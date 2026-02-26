"""
src/agent.py

Main entry-point for CommunityPulse AI Agent.

Usage (CLI):
    python -m src.agent --community seattle portland denver --offline

Usage (Python):
    from src.agent import CommunityPulseRunner
    runner = CommunityPulseRunner(offline=True)
    runner.run_analysis()
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import CommunityPulseDataLoader
from src.models import BERTEmbedder, CommunityPulseAgent, TopicModeler
from src.analysis import CommunityPulseAnalyser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class CommunityPulseRunner:
    """
    Orchestrates the full CommunityPulse pipeline:

      1. Load three data modalities (Reddit, Census, News).
      2. Embed Reddit posts with BERT (Week 1 model).
      3. Fit a topic model on the text corpus (Week 2 model).
      4. Run the LangChain LLM agent for synthesis (Week 3 model).
      5. Produce qualitative validation summaries.

    Parameters
    ----------
    communities : list[str], optional
        City names to analyse.
    offline : bool
        Use bundled sample data and skip API calls.
    openai_api_key : str, optional
        OpenAI key; falls back to OPENAI_API_KEY env var.
    reddit_credentials : dict, optional
        PRAW credentials (client_id, client_secret, user_agent).
    census_api_key : str, optional
        US Census API key.
    """

    DEFAULT_COMMUNITIES = ["seattle", "portland", "denver"]

    def __init__(
        self,
        communities: list[str] | None = None,
        offline: bool = True,
        openai_api_key: str | None = None,
        reddit_credentials: dict | None = None,
        census_api_key: str | None = None,
    ) -> None:
        self.communities = communities or self.DEFAULT_COMMUNITIES
        self.offline = offline
        self.openai_api_key = openai_api_key
        self.reddit_credentials = reddit_credentials
        self.census_api_key = census_api_key

        # Initialised lazily
        self.data: dict[str, pd.DataFrame] | None = None
        self.embedder: BERTEmbedder | None = None
        self.topic_modeler: TopicModeler | None = None
        self.agent: CommunityPulseAgent | None = None
        self.analyser: CommunityPulseAnalyser | None = None

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def step_load_data(self) -> dict[str, pd.DataFrame]:
        """Step 1 – Load all three data modalities."""
        logger.info("=== Step 1: Loading data ===")
        loader = CommunityPulseDataLoader(
            communities=self.communities,
            offline=self.offline,
            reddit_credentials=self.reddit_credentials,
            census_api_key=self.census_api_key,
        )
        self.data = loader.load_all()
        return self.data

    def step_embed_text(self) -> pd.DataFrame:
        """Step 2 – Embed Reddit posts with BERT (Week 1 model)."""
        logger.info("=== Step 2: BERT Embeddings ===")
        self.embedder = BERTEmbedder()
        self.data["reddit"] = self.embedder.embed_dataframe(self.data["reddit"], text_col="title")
        logger.info("Embedded %d Reddit posts.", len(self.data["reddit"]))
        return self.data["reddit"]

    def step_fit_topics(self) -> pd.DataFrame:
        """Step 3 – Fit topic model on Reddit text (Week 2 model)."""
        logger.info("=== Step 3: Topic Modelling ===")
        self.topic_modeler = TopicModeler(n_topics=8, min_topic_size=2)
        df = self.data["reddit"]
        docs = (df["title"].fillna("") + " " + df["selftext"].fillna("")).tolist()
        self.topic_modeler.fit(docs)
        assignments = self.topic_modeler.assign_topics(docs)
        self.data["reddit"]["topic"] = assignments
        topic_table = self.topic_modeler.topic_summary_table()
        logger.info("Topic summary:\n%s", topic_table.to_string(index=False))
        return topic_table

    def step_run_agent(self, query: str | None = None) -> str:
        """Step 4 – Run LangChain agent for synthesis (Week 3 model)."""
        logger.info("=== Step 4: LLM Agent Synthesis ===")
        self.agent = CommunityPulseAgent(
            embedder=self.embedder,
            topic_modeler=self.topic_modeler,
            data=self.data,
            openai_api_key=self.openai_api_key,
            model_name="gpt-3.5-turbo",
        )
        default_query = (
            "Compare the social dynamics of "
            + ", ".join(c.title() for c in self.communities)
            + " based on news sentiment, Reddit discussion topics, and demographic data. "
            "Identify which community faces the most acute social challenges and why."
        )
        result = self.agent.run(query or default_query)
        logger.info("Agent output:\n%s", result)
        return result

    def step_qualitative_validation(self) -> pd.DataFrame:
        """Step 5 – Produce qualitative validation summaries."""
        logger.info("=== Step 5: Qualitative Validation ===")
        self.analyser = CommunityPulseAnalyser(
            data=self.data,
            topic_modeler=self.topic_modeler,
        )
        summary = self.analyser.community_summary_table()
        logger.info("Validation summary:\n%s", summary.to_string(index=False))
        return summary

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_analysis(self, agent_query: str | None = None) -> dict:
        """Run the complete pipeline end-to-end."""
        self.step_load_data()
        self.step_embed_text()
        self.step_fit_topics()
        agent_output = self.step_run_agent(agent_query)
        validation = self.step_qualitative_validation()
        return {
            "data": self.data,
            "agent_output": agent_output,
            "validation_table": validation,
        }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="community_pulse",
        description="CommunityPulse AI Agent – multi-modal urban social analysis",
    )
    parser.add_argument(
        "--community",
        nargs="+",
        default=["seattle", "portland", "denver"],
        help="City names to analyse (default: seattle portland denver)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Use bundled sample data (no API keys required)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Attempt live API calls (requires credentials in environment)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom natural-language query for the LLM agent",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    offline = not args.live

    runner = CommunityPulseRunner(
        communities=args.community,
        offline=offline,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        reddit_credentials={
            "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
            "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
            "user_agent": os.getenv("REDDIT_USER_AGENT", "CommunityPulse/1.0"),
        },
        census_api_key=os.getenv("CENSUS_API_KEY"),
    )

    results = runner.run_analysis(agent_query=args.query)

    print("\n" + "=" * 60)
    print("AGENT OUTPUT")
    print("=" * 60)
    print(results["agent_output"])
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(results["validation_table"].to_string(index=False))


if __name__ == "__main__":
    main()
