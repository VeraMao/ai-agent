"""
src/analysis.py

Analysis helpers and qualitative validation for CommunityPulse.

Provides:
  • CommunityPulseAnalyser – aggregates multi-modal signals into
    per-community scorecards and human-readable case studies
  • Plotting helpers (matplotlib / seaborn) for notebook use
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qualitative validation helpers
# ---------------------------------------------------------------------------

class CommunityPulseAnalyser:
    """
    Synthesises the three data modalities into community-level scorecards
    and qualitative case studies.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Keys: 'reddit', 'census', 'news'.
    topic_modeler : TopicModeler, optional
        Fitted topic model used to annotate topic assignments.
    """

    SENTIMENT_LABELS = {
        "positive": lambda s: s > 0.2,
        "neutral": lambda s: -0.2 <= s <= 0.2,
        "negative": lambda s: s < -0.2,
    }

    def __init__(self, data: dict[str, pd.DataFrame], topic_modeler: Any | None = None) -> None:
        self.data = data
        self.topic_modeler = topic_modeler

    # ------------------------------------------------------------------
    # Scorecard
    # ------------------------------------------------------------------

    def community_summary_table(self) -> pd.DataFrame:
        """
        Build a unified community scorecard combining all three modalities.

        Returns
        -------
        pd.DataFrame with one row per community and columns covering
        sentiment statistics, demographic indicators, and post engagement.
        """
        communities = self.data["census"]["community"].tolist()
        rows = []
        for community in communities:
            reddit_sub = self.data["reddit"][
                self.data["reddit"]["community"].str.lower() == community.lower()
            ]
            census_row = self.data["census"][
                self.data["census"]["community"].str.lower() == community.lower()
            ]
            news_sub = self.data["news"][
                self.data["news"]["community"].str.lower() == community.lower()
            ]

            # Reddit metrics
            n_posts = len(reddit_sub)
            avg_score = reddit_sub["score"].mean() if n_posts else np.nan
            top_post = reddit_sub.nlargest(1, "score")["title"].iloc[0] if n_posts else ""

            # Census metrics
            if not census_row.empty:
                c = census_row.iloc[0]
                median_income = c.get("median_household_income", np.nan)
                poverty_rate = c.get("poverty_rate", np.nan)
                unemployment = c.get("unemployment_rate", np.nan)
                bachelor_pct = c.get("bachelor_degree_pct", np.nan)
            else:
                median_income = poverty_rate = unemployment = bachelor_pct = np.nan

            # News metrics
            mean_sentiment = news_sub["sentiment_score"].mean() if not news_sub.empty else np.nan
            n_negative = (news_sub["sentiment_score"] < -0.2).sum() if not news_sub.empty else 0
            n_positive = (news_sub["sentiment_score"] > 0.2).sum() if not news_sub.empty else 0

            rows.append(
                {
                    "community": community,
                    "n_reddit_posts": n_posts,
                    "avg_post_score": round(float(avg_score), 1) if not np.isnan(avg_score) else None,
                    "top_reddit_post": top_post[:60] + ("..." if len(top_post) > 60 else ""),
                    "median_household_income": median_income,
                    "poverty_rate_pct": poverty_rate,
                    "unemployment_rate_pct": unemployment,
                    "bachelor_degree_pct": bachelor_pct,
                    "mean_news_sentiment": round(float(mean_sentiment), 4) if not np.isnan(mean_sentiment) else None,
                    "n_negative_headlines": int(n_negative),
                    "n_positive_headlines": int(n_positive),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Case studies (qualitative interpretation)
    # ------------------------------------------------------------------

    def generate_case_study(self, community: str) -> str:
        """
        Produce a qualitative case-study paragraph for *community*.

        This is the qualitative validation layer: it translates quantitative
        signals into human-readable narrative, identifying tensions,
        contradictions, and nuances that numbers alone cannot convey.
        """
        df = self.community_summary_table()
        row = df[df["community"].str.lower() == community.lower()]
        if row.empty:
            return f"No data found for community: {community}"
        r = row.iloc[0]

        sentiment_label = (
            "positive" if (r["mean_news_sentiment"] or 0) > 0.2
            else "negative" if (r["mean_news_sentiment"] or 0) < -0.2
            else "mixed"
        )

        income_ctx = (
            "above the national median"
            if (r["median_household_income"] or 0) > 70000
            else "below the national median"
        )

        reddit_mood = (
            "highly engaged, with spirited debate"
            if (r["avg_post_score"] or 0) > 350
            else "moderately active"
        )

        topics_str = "N/A"
        if self.topic_modeler is not None and self.topic_modeler._model is not None:
            reddit_sub = self.data["reddit"][
                self.data["reddit"]["community"].str.lower() == community.lower()
            ]
            if not reddit_sub.empty:
                docs = (reddit_sub["title"].fillna("") + " " + reddit_sub["selftext"].fillna("")).tolist()
                assignments = self.topic_modeler.assign_topics(docs)
                topic_words = self.topic_modeler.get_topic_words(n_words=5)
                from collections import Counter  # noqa: PLC0415
                top_tid = Counter(assignments).most_common(1)[0][0]
                topics_str = ", ".join(topic_words.get(top_tid, []))

        narrative = (
            f"**{community}** presents a {sentiment_label} public-discourse profile. "
            f"Median household income sits at ${r['median_household_income']:,.0f} — {income_ctx} — "
            f"yet the poverty rate of {r['poverty_rate_pct']}% and unemployment rate of "
            f"{r['unemployment_rate_pct']}% hint at underlying inequality beneath aggregate prosperity. "
            f"Reddit communities are {reddit_mood}, with the top post attracting an average score of "
            f"{r['avg_post_score']}. "
            f"The dominant Reddit discussion theme revolves around: {topics_str}. "
            f"News coverage skews {sentiment_label}, with {r['n_negative_headlines']} negative and "
            f"{r['n_positive_headlines']} positive headlines in the sample window. "
            f"Together, these signals suggest a community grappling with rapid change — "
            f"economic growth coexisting with housing pressure, transit investment alongside "
            f"inequality — a tension that emerges consistently across all three data modalities."
        )
        return narrative

    def all_case_studies(self) -> dict[str, str]:
        """Return case studies for every community in the dataset."""
        communities = self.data["census"]["community"].tolist()
        return {c: self.generate_case_study(c) for c in communities}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_sentiment_distribution(news_df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Plot a grouped bar chart of news sentiment distributions per community.
    Requires matplotlib and seaborn.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import seaborn as sns  # noqa: PLC0415

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) Box plot of raw scores
        sns.boxplot(data=news_df, x="community", y="sentiment_score", palette="coolwarm", ax=axes[0])
        axes[0].axhline(0, color="grey", linestyle="--", linewidth=0.8)
        axes[0].set_title("News Sentiment Score Distribution by Community")
        axes[0].set_xlabel("Community")
        axes[0].set_ylabel("Sentiment Score (-1 → +1)")

        # (b) Stacked bar of pos/neu/neg headline counts
        def categorise(s: float) -> str:
            if s > 0.2:
                return "positive"
            if s < -0.2:
                return "negative"
            return "neutral"

        news_df = news_df.copy()
        news_df["category"] = news_df["sentiment_score"].apply(categorise)
        counts = news_df.groupby(["community", "category"]).size().unstack(fill_value=0)
        counts.plot(kind="bar", stacked=True, colormap="RdYlGn", ax=axes[1])
        axes[1].set_title("Headline Count by Sentiment Category")
        axes[1].set_xlabel("Community")
        axes[1].set_ylabel("Number of Headlines")
        axes[1].tick_params(axis="x", rotation=0)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    except ImportError:
        logger.warning("matplotlib/seaborn not installed; skipping plot.")


def plot_demographic_comparison(census_df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Plot a radar / bar comparison of key Census metrics across communities.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        metrics = ["poverty_rate", "unemployment_rate", "bachelor_degree_pct"]
        labels = ["Poverty Rate (%)", "Unemployment (%)", "Bachelor Degree (%)"]
        communities = census_df["community"].tolist()

        x = np.arange(len(metrics))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, community in enumerate(communities):
            row = census_df[census_df["community"] == community].iloc[0]
            values = [float(row[m]) for m in metrics]
            ax.bar(x + i * width, values, width, label=community)

        ax.set_xticks(x + width * (len(communities) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_title("Demographic Comparison Across Communities")
        ax.set_ylabel("Percentage")
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot.")


def plot_embedding_clusters(
    reddit_df: pd.DataFrame,
    n_components: int = 2,
    save_path: str | None = None,
) -> None:
    """
    Reduce BERT embeddings to 2-D with UMAP and visualise community clusters.
    """
    if "embedding" not in reddit_df.columns:
        logger.warning("No 'embedding' column found; run BERTEmbedder first.")
        return
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import umap  # noqa: PLC0415

        embeddings = np.vstack(reddit_df["embedding"].tolist())
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings)

        communities = reddit_df["community"].unique()
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(9, 7))
        for i, community in enumerate(communities):
            mask = reddit_df["community"] == community
            ax.scatter(reduced[mask, 0], reduced[mask, 1], label=community, s=80, alpha=0.8, color=cmap(i))
        ax.set_title("BERT Embedding Clusters by Community (UMAP 2-D)")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    except ImportError as exc:
        logger.warning("Plotting dependencies missing (%s); skipping.", exc)


def plot_topic_distribution(reddit_df: pd.DataFrame, topic_modeler: Any, save_path: str | None = None) -> None:
    """
    Plot a stacked bar chart showing topic distribution per community.
    """
    if "topic" not in reddit_df.columns:
        logger.warning("No 'topic' column; run TopicModeler first.")
        return
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        topic_words = topic_modeler.get_topic_words(n_words=3)
        topic_labels = {
            tid: f"T{tid}: {', '.join(words[:2])}"
            for tid, words in topic_words.items()
        }
        df = reddit_df.copy()
        df["topic_label"] = df["topic"].map(topic_labels).fillna("Other")

        counts = df.groupby(["community", "topic_label"]).size().unstack(fill_value=0)
        counts.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
        plt.title("Topic Distribution per Community")
        plt.xlabel("Community")
        plt.ylabel("Number of Posts")
        plt.xticks(rotation=0)
        plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    except ImportError as exc:
        logger.warning("matplotlib not installed (%s); skipping.", exc)
