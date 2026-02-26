"""
src/models.py

Three model families used in CommunityPulse:

  Week 1 – BERT Sentence Embeddings (transformer NLP)
    Encodes Reddit post titles + body into dense 384-d vectors using
    sentence-transformers ("sentence-transformers/all-MiniLM-L6-v2").

  Week 2 – BERTopic / LDA Topic Model (probabilistic unsupervised learning)
    Discovers latent topics across all community posts.
    Falls back to scikit-learn LatentDirichletAllocation when BERTopic
    or its dependencies are not installed.

  Week 3 – LangChain Tool-Calling LLM Agent
    Wraps the two upstream models as LangChain tools so a GPT-4 / GPT-3.5
    agent can reason over multi-modal evidence, compare communities, and
    generate qualitative summaries.  Falls back to a lightweight offline
    stub when no API key is available.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Week 1 – BERT Sentence Embeddings
# ---------------------------------------------------------------------------

class BERTEmbedder:
    """
    Encodes text into dense sentence embeddings using sentence-transformers.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to the lightweight
        'sentence-transformers/all-MiniLM-L6-v2' (80 MB, 384-d embeddings).
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # lazy-load

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                self._model = SentenceTransformer(self.model_name)
                logger.info("BERTEmbedder: loaded model '%s'", self.model_name)
            except Exception as exc:
                logger.warning("BERTEmbedder: could not load model (%s). Using random embeddings.", exc)
                self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray of shape (n_texts, embedding_dim)
        """
        self._ensure_loaded()
        if self._model is not None:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        # Offline fallback: deterministic pseudo-embeddings based on text hash
        return np.array(
            [np.sin(np.arange(384) * (hash(t) % 1000 + 1) / 1000.0) for t in texts],
            dtype=np.float32,
        )

    def embed_dataframe(self, df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
        """
        Add an 'embedding' column (list of floats) to *df* and return it.
        """
        texts = (df[text_col].fillna("") + " " + df.get("selftext", pd.Series([""] * len(df))).fillna("")).tolist()
        embeddings = self.embed(texts)
        df = df.copy()
        df["embedding"] = list(embeddings)
        return df


# ---------------------------------------------------------------------------
# Week 2 – Topic Model (BERTopic with LDA fallback)
# ---------------------------------------------------------------------------

class TopicModeler:
    """
    Fits a topic model on a corpus of documents.

    Tries BERTopic first; falls back to scikit-learn LDA if BERTopic or
    its UMAP/HDBSCAN dependencies are unavailable.

    Parameters
    ----------
    n_topics : int
        Number of topics (used by LDA fallback).  BERTopic auto-detects.
    min_topic_size : int
        Minimum cluster size for BERTopic.
    """

    def __init__(self, n_topics: int = 8, min_topic_size: int = 2) -> None:
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self._model = None
        self._vectorizer = None
        self._backend = None  # "bertopic" or "lda"

    # ------------------------------------------------------------------
    def fit(self, documents: list[str]) -> "TopicModeler":
        """Fit the topic model on *documents*."""
        try:
            self._fit_bertopic(documents)
        except Exception as exc:
            logger.warning("BERTopic fit failed (%s); falling back to LDA.", exc)
            self._fit_lda(documents)
        return self

    def _fit_bertopic(self, documents: list[str]) -> None:
        from bertopic import BERTopic  # noqa: PLC0415

        self._model = BERTopic(min_topic_size=self.min_topic_size, verbose=False)
        self._topics, _ = self._model.fit_transform(documents)
        self._backend = "bertopic"
        logger.info("TopicModeler: BERTopic fitted, %d topics found.", len(set(self._topics)) - 1)

    def _fit_lda(self, documents: list[str]) -> None:
        from sklearn.feature_extraction.text import CountVectorizer  # noqa: PLC0415
        from sklearn.decomposition import LatentDirichletAllocation  # noqa: PLC0415

        self._vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words="english")
        dtm = self._vectorizer.fit_transform(documents)
        self._model = LatentDirichletAllocation(
            n_components=min(self.n_topics, max(2, len(documents) // 2)),
            random_state=42,
            max_iter=20,
        )
        self._model.fit(dtm)
        self._topics = self._model.transform(dtm).argmax(axis=1).tolist()
        self._backend = "lda"
        logger.info("TopicModeler: LDA fitted with %d topics.", self._model.n_components)

    # ------------------------------------------------------------------
    def get_topic_words(self, n_words: int = 10) -> dict[int, list[str]]:
        """
        Return the top-n words for each topic.

        Returns
        -------
        dict mapping topic_id → list of keyword strings
        """
        if self._backend == "bertopic":
            info = self._model.get_topic_info()
            result = {}
            for _, row in info.iterrows():
                tid = row["Topic"]
                if tid == -1:
                    continue
                words = [w for w, _ in self._model.get_topic(tid)[:n_words]]
                result[tid] = words
            return result
        # LDA
        feature_names = self._vectorizer.get_feature_names_out()
        result = {}
        for idx, topic in enumerate(self._model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            result[idx] = [feature_names[i] for i in top_indices]
        return result

    def assign_topics(self, documents: list[str]) -> list[int]:
        """Return topic assignment for each document."""
        if self._backend == "bertopic":
            topics, _ = self._model.transform(documents)
            return topics
        dtm = self._vectorizer.transform(documents)
        return self._model.transform(dtm).argmax(axis=1).tolist()

    def topic_summary_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising each topic."""
        topic_words = self.get_topic_words()
        rows = [
            {"topic_id": tid, "top_keywords": ", ".join(words)}
            for tid, words in sorted(topic_words.items())
        ]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Week 3 – LangChain Tool-Calling LLM Agent
# ---------------------------------------------------------------------------

class CommunityPulseAgent:
    """
    A LangChain agent that orchestrates multi-modal analysis via tools.

    Tools available to the agent:
      • get_community_sentiment  – average news sentiment for a community
      • get_top_topics           – dominant topics from Reddit posts
      • get_demographic_profile  – Census summary for a community
      • compare_communities      – side-by-side comparison across metrics

    Falls back to a deterministic offline stub when OPENAI_API_KEY is absent.

    Parameters
    ----------
    embedder : BERTEmbedder
    topic_modeler : TopicModeler
    data : dict[str, pd.DataFrame]
        Pre-loaded dict with keys 'reddit', 'census', 'news'.
    openai_api_key : str, optional
        OpenAI key.  If None, the env var OPENAI_API_KEY is used.
    model_name : str
        OpenAI chat model to use.
    """

    def __init__(
        self,
        embedder: BERTEmbedder,
        topic_modeler: TopicModeler,
        data: dict[str, pd.DataFrame],
        openai_api_key: str | None = None,
        model_name: str = "gpt-3.5-turbo",
    ) -> None:
        self.embedder = embedder
        self.topic_modeler = topic_modeler
        self.data = data
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name
        self._chain = None

    # ------------------------------------------------------------------
    # Tool implementations (called by the agent or directly)
    # ------------------------------------------------------------------

    def get_community_sentiment(self, community: str) -> dict:
        """Return average and std of news sentiment for a community."""
        df = self.data["news"]
        subset = df[df["community"].str.lower() == community.lower()]
        if subset.empty:
            return {"community": community, "mean_sentiment": None, "std_sentiment": None, "n_articles": 0}
        return {
            "community": community,
            "mean_sentiment": round(float(subset["sentiment_score"].mean()), 4),
            "std_sentiment": round(float(subset["sentiment_score"].std()), 4),
            "n_articles": len(subset),
        }

    def get_top_topics(self, community: str, n: int = 3) -> dict:
        """Return the top-n topics for a community based on Reddit posts."""
        df = self.data["reddit"]
        subset = df[df["community"].str.lower() == community.lower()]
        if subset.empty or self.topic_modeler._model is None:
            return {"community": community, "topics": []}
        docs = (subset["title"].fillna("") + " " + subset["selftext"].fillna("")).tolist()
        assignments = self.topic_modeler.assign_topics(docs)
        from collections import Counter  # noqa: PLC0415

        top = Counter(assignments).most_common(n)
        topic_words = self.topic_modeler.get_topic_words()
        return {
            "community": community,
            "topics": [
                {"topic_id": tid, "count": cnt, "keywords": topic_words.get(tid, [])}
                for tid, cnt in top
            ],
        }

    def get_demographic_profile(self, community: str) -> dict:
        """Return Census demographic summary for a community."""
        df = self.data["census"]
        row = df[df["community"].str.lower() == community.lower()]
        if row.empty:
            return {"community": community, "profile": None}
        record = row.iloc[0].to_dict()
        return {"community": community, "profile": record}

    def compare_communities(self, communities: list[str]) -> pd.DataFrame:
        """Side-by-side comparison of sentiment, topics, and demographics."""
        rows = []
        for c in communities:
            sentiment = self.get_community_sentiment(c)
            demo = self.get_demographic_profile(c)
            profile = demo.get("profile") or {}
            rows.append(
                {
                    "community": c,
                    "mean_sentiment": sentiment.get("mean_sentiment"),
                    "median_income": profile.get("median_household_income"),
                    "poverty_rate": profile.get("poverty_rate"),
                    "bachelor_pct": profile.get("bachelor_degree_pct"),
                    "unemployment_rate": profile.get("unemployment_rate"),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _build_langchain_agent(self):
        """Construct and cache the LangChain agent."""
        if self._chain is not None:
            return self._chain

        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        from langchain.agents import AgentExecutor, create_tool_calling_agent  # noqa: PLC0415
        from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
        from langchain_core.tools import tool  # noqa: PLC0415
        import json  # noqa: PLC0415

        agent_self = self  # capture for closures

        @tool
        def community_sentiment(community: str) -> str:
            """Get average news sentiment score for a given community (city name)."""
            return json.dumps(agent_self.get_community_sentiment(community))

        @tool
        def top_topics(community: str) -> str:
            """Get top Reddit discussion topics for a given community (city name)."""
            return json.dumps(agent_self.get_top_topics(community))

        @tool
        def demographic_profile(community: str) -> str:
            """Get Census demographic profile for a given community (city name)."""
            return json.dumps(agent_self.get_demographic_profile(community))

        tools = [community_sentiment, top_topics, demographic_profile]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are CommunityPulse, an AI analyst specialising in urban social dynamics. "
                    "Use your tools to gather multi-modal evidence (news sentiment, Reddit topics, "
                    "Census demographics) and synthesise insightful, nuanced summaries. "
                    "Always cite the data you used and acknowledge uncertainty.",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.3,
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        self._chain = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return self._chain

    def run(self, query: str) -> str:
        """
        Execute a natural-language query against the multi-modal data.

        Falls back to an offline deterministic answer when no API key is set.
        """
        if not self.api_key:
            return self._offline_run(query)
        try:
            chain = self._build_langchain_agent()
            result = chain.invoke({"input": query})
            return result.get("output", str(result))
        except Exception as exc:
            logger.warning("LangChain agent failed (%s); falling back to offline mode.", exc)
            return self._offline_run(query)

    def _offline_run(self, query: str) -> str:
        """
        Offline stub: assemble a structured answer from tool outputs.
        Used when no OpenAI API key is available.
        """
        communities = [
            c for c in self.data["census"]["community"].tolist()
            if c.lower() in query.lower()
        ]
        if not communities:
            communities = self.data["census"]["community"].tolist()

        lines = [f"[Offline mode] Analysing {len(communities)} communities for: '{query}'\n"]
        for c in communities:
            sentiment = self.get_community_sentiment(c)
            topics = self.get_top_topics(c)
            demo = self.get_demographic_profile(c)
            profile = demo.get("profile") or {}

            lines.append(f"### {c}")
            lines.append(
                f"  News sentiment: mean={sentiment['mean_sentiment']}, "
                f"std={sentiment['std_sentiment']} (n={sentiment['n_articles']})"
            )
            top_kw = []
            for t in topics.get("topics", []):
                top_kw.extend(t.get("keywords", [])[:3])
            lines.append(f"  Top Reddit keywords: {', '.join(top_kw[:9]) or 'N/A'}")
            lines.append(
                f"  Median income: ${profile.get('median_household_income', 'N/A'):,}  "
                f"Poverty rate: {profile.get('poverty_rate', 'N/A')}%  "
                f"Unemployment: {profile.get('unemployment_rate', 'N/A')}%"
            )
            lines.append("")
        return "\n".join(lines)
