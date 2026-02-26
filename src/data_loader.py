"""
src/data_loader.py

Loaders for the three data modalities used in CommunityPulse:

  1. Reddit posts  – text (PRAW or offline sample CSV)
  2. Census ACS    – structured tabular (census Python client or sample CSV)
  3. News headlines – temporal time-series (RSS feeds or sample CSV)

All loaders return a pandas DataFrame with a consistent schema so the rest of
the pipeline can treat each modality uniformly.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample data helpers (used when offline=True or credentials not supplied)
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"

_SAMPLE_REDDIT = """subreddit,community,post_id,title,selftext,score,created_utc
seattle,Seattle,t3_a1,High rent forcing long-time residents out of Capitol Hill,Landlord raised my rent 40% this year. Lived here 8 years.,412,2024-01-15T10:23:00
seattle,Seattle,t3_a2,Homelessness crisis downtown – city council meeting recap,Good turnout at the council meeting. Mixed feelings on the new shelter plan.,289,2024-01-16T14:05:00
seattle,Seattle,t3_a3,New light-rail extension opens next month – what do you think?,Finally! This will cut my commute in half. Can't wait.,534,2024-01-17T09:11:00
seattle,Seattle,t3_a4,Community garden in Rainier Beach looking for volunteers,We need weekend help planting and watering. All ages welcome.,198,2024-01-18T08:00:00
seattle,Seattle,t3_a5,Tech layoffs hitting Seattle hard – anyone else seeing this?,Got laid off from Amazon last week. Scary times.,671,2024-01-19T11:30:00
portland,Portland,t3_b1,Wildfires and air quality – how are you coping?,AQI hit 180 today. Kids can't play outside.,305,2024-01-15T09:00:00
portland,Portland,t3_b2,New affordable housing development approved in Lents,Great news for the neighborhood! Long overdue.,412,2024-01-16T13:15:00
portland,Portland,t3_b3,Bike lane expansion causes controversy on Division St,I love it but the small businesses are struggling.,178,2024-01-17T10:45:00
portland,Portland,t3_b4,Food cart pod closing after 10 years – sad day,Best Korean BBQ in the city. This is a real loss.,289,2024-01-18T16:20:00
portland,Portland,t3_b5,MAX train delays are getting worse – daily rant,Three delays this week alone. TriMet needs to fix this.,234,2024-01-19T07:55:00
denver,Denver,t3_c1,Altitude sickness in new residents – tips and tricks,Moved from sea level. First month was rough.,156,2024-01-15T11:00:00
denver,Denver,t3_c2,Gentrification in Five Points – community perspectives,My family has been here 30 years. The neighborhood is unrecognizable.,489,2024-01-16T10:30:00
denver,Denver,t3_c3,Ski season traffic on I-70 is unbearable again,Three hours to get to Breckenridge on Saturday. Never again.,312,2024-01-17T08:20:00
denver,Denver,t3_c4,Local brewery wins national award – Denver pride!,Ratio Beerworks is the best. So well deserved.,267,2024-01-18T17:00:00
denver,Denver,t3_c5,Marijuana dispensary saturation – too many or just right?,There's one on every block now. Mixed feelings.,198,2024-01-19T12:10:00
"""

_SAMPLE_CENSUS = """community,state,county_fips,median_household_income,median_rent,poverty_rate,total_population,white_pct,black_pct,hispanic_pct,asian_pct,bachelor_degree_pct,unemployment_rate
Seattle,WA,53033,105391,1832,10.2,749256,65.3,7.1,7.0,16.0,62.4,3.1
Portland,OR,41051,78439,1412,13.5,652503,72.1,5.9,10.2,8.2,47.8,4.2
Denver,CO,08031,68592,1356,13.9,715522,68.4,9.7,29.8,4.1,50.3,4.8
"""

_SAMPLE_NEWS = """community,headline,source,published_date,sentiment_score
Seattle,Amazon announces new downtown HQ expansion,Seattle Times,2024-01-10,0.65
Seattle,Seattle homeless count rises 5% year-over-year,Seattle PI,2024-01-11,-0.48
Seattle,City approves record affordable housing budget,KUOW,2024-01-12,0.72
Seattle,Seattle named top tech job market for third year,GeekWire,2024-01-13,0.81
Seattle,Fentanyl crisis claims 12 lives in one weekend,KOMO News,2024-01-14,-0.89
Portland,Portland declares state of emergency over fentanyl,OregonLive,2024-01-10,-0.92
Portland,New MAX line opening boosts transit ridership,Portland Tribune,2024-01-11,0.61
Portland,Small business closures accelerate in Old Town,Willamette Week,2024-01-12,-0.55
Portland,Portland food scene earns James Beard nominations,Portland Monthly,2024-01-13,0.78
Portland,Protest erupts over new homeless campsite policy,KGW,2024-01-14,-0.44
Denver,Denver economy outpaces national growth,Denver Post,2024-01-10,0.74
Denver,Colorado ski industry reports record season,Summit Daily,2024-01-11,0.83
Denver,Denver school district faces budget shortfall,Denver7,2024-01-12,-0.61
Denver,New light rail extension opens on time and on budget,CPR News,2024-01-13,0.69
Denver,Surge in property crime reported in Capitol Hill,KDVR,2024-01-14,-0.57
"""


def _write_sample_csv(name: str, content: str) -> Path:
    """Write a sample CSV to data/sample/ if it doesn't already exist."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_DIR / name
    if not path.exists():
        path.write_text(content.strip())
    return path


# ---------------------------------------------------------------------------
# Modality 1 – Reddit (text)
# ---------------------------------------------------------------------------

class RedditLoader:
    """
    Load Reddit posts either via PRAW (live) or from a local sample CSV.

    Parameters
    ----------
    communities : list[str]
        Subreddit names to query (e.g. ["seattle", "portland"]).
    limit : int
        Maximum number of posts per subreddit (live mode only).
    offline : bool
        When True, always use the bundled sample CSV.
    reddit_credentials : dict, optional
        Keys: client_id, client_secret, user_agent.
    """

    SCHEMA = ["subreddit", "community", "post_id", "title", "selftext", "score", "created_utc"]

    def __init__(
        self,
        communities: list[str] | None = None,
        limit: int = 100,
        offline: bool = True,
        reddit_credentials: dict | None = None,
    ) -> None:
        self.communities = communities or ["seattle", "portland", "denver"]
        self.limit = limit
        self.offline = offline
        self.credentials = reddit_credentials or {}

    def load(self) -> pd.DataFrame:
        """Return a DataFrame with Reddit posts."""
        if self.offline or not self.credentials:
            return self._load_sample()
        return self._load_live()

    def _load_sample(self) -> pd.DataFrame:
        path = _write_sample_csv("reddit_sample.csv", _SAMPLE_REDDIT)
        df = pd.read_csv(path, parse_dates=["created_utc"])
        df = df[df["subreddit"].isin(self.communities)]
        logger.info("RedditLoader: loaded %d posts from sample CSV", len(df))
        return df.reset_index(drop=True)

    def _load_live(self) -> pd.DataFrame:
        try:
            import praw  # noqa: PLC0415

            reddit = praw.Reddit(
                client_id=self.credentials.get("client_id"),
                client_secret=self.credentials.get("client_secret"),
                user_agent=self.credentials.get("user_agent", "CommunityPulse/1.0"),
            )
            rows = []
            for sub in self.communities:
                for post in reddit.subreddit(sub).hot(limit=self.limit):
                    rows.append(
                        {
                            "subreddit": sub,
                            "community": sub.title(),
                            "post_id": post.id,
                            "title": post.title,
                            "selftext": post.selftext,
                            "score": post.score,
                            "created_utc": datetime.utcfromtimestamp(post.created_utc),
                        }
                    )
            df = pd.DataFrame(rows, columns=self.SCHEMA)
            logger.info("RedditLoader: fetched %d live posts", len(df))
            return df
        except Exception as exc:
            logger.warning("RedditLoader live fetch failed (%s); falling back to sample.", exc)
            return self._load_sample()


# ---------------------------------------------------------------------------
# Modality 2 – Census ACS (structured tabular)
# ---------------------------------------------------------------------------

class CensusLoader:
    """
    Load US Census American Community Survey (ACS) data.

    Parameters
    ----------
    communities : list[str]
        City names matching the sample CSV (Seattle, Portland, Denver).
    offline : bool
        When True, always use the bundled sample CSV.
    census_api_key : str, optional
        Census Bureau API key for live queries.
    """

    NUMERIC_COLS = [
        "median_household_income",
        "median_rent",
        "poverty_rate",
        "total_population",
        "white_pct",
        "black_pct",
        "hispanic_pct",
        "asian_pct",
        "bachelor_degree_pct",
        "unemployment_rate",
    ]

    def __init__(
        self,
        communities: list[str] | None = None,
        offline: bool = True,
        census_api_key: str | None = None,
    ) -> None:
        self.communities = [c.title() for c in (communities or ["seattle", "portland", "denver"])]
        self.offline = offline
        self.api_key = census_api_key

    def load(self) -> pd.DataFrame:
        """Return a DataFrame with one row per community."""
        if self.offline or not self.api_key:
            return self._load_sample()
        return self._load_live()

    def _load_sample(self) -> pd.DataFrame:
        path = _write_sample_csv("census_sample.csv", _SAMPLE_CENSUS)
        df = pd.read_csv(path)
        df = df[df["community"].isin(self.communities)]
        logger.info("CensusLoader: loaded %d rows from sample CSV", len(df))
        return df.reset_index(drop=True)

    def _load_live(self) -> pd.DataFrame:
        try:
            from census import Census  # noqa: PLC0415
            import us  # noqa: PLC0415

            c = Census(self.api_key)
            variables = {
                "B19013_001E": "median_household_income",
                "B25064_001E": "median_rent",
                "B17001_002E": "poverty_count",
                "B01003_001E": "total_population",
            }
            rows = []
            for community in self.communities:
                state_obj = us.states.lookup(community)
                if state_obj is None:
                    continue
                data = c.acs5.state_county(
                    list(variables.keys()),
                    state_obj.fips,
                    Census.ALL,
                )
                for record in data:
                    row = {"community": community}
                    for api_key_name, col in variables.items():
                        row[col] = record.get(api_key_name)
                    rows.append(row)
            df = pd.DataFrame(rows)
            logger.info("CensusLoader: fetched %d live rows", len(df))
            return df
        except Exception as exc:
            logger.warning("CensusLoader live fetch failed (%s); falling back to sample.", exc)
            return self._load_sample()


# ---------------------------------------------------------------------------
# Modality 3 – News headlines (temporal / time-series)
# ---------------------------------------------------------------------------

class NewsLoader:
    """
    Load news headlines with pre-computed sentiment scores.

    In offline mode, uses the bundled sample CSV.
    In live mode, fetches RSS feeds and computes sentiment with
    a lightweight VADER-based scorer.

    Parameters
    ----------
    communities : list[str]
        Community names to include.
    start_date : str, optional
        ISO date string for the earliest article (live mode).
    offline : bool
        When True, always use the bundled sample CSV.
    rss_feeds : dict[str, list[str]], optional
        Mapping community → list of RSS feed URLs.
    """

    def __init__(
        self,
        communities: list[str] | None = None,
        start_date: str | None = None,
        offline: bool = True,
        rss_feeds: dict | None = None,
    ) -> None:
        self.communities = [c.title() for c in (communities or ["seattle", "portland", "denver"])]
        self.start_date = start_date or (datetime.utcnow() - timedelta(days=30)).isoformat()
        self.offline = offline
        self.rss_feeds = rss_feeds or {}

    def load(self) -> pd.DataFrame:
        """Return a DataFrame with one row per headline."""
        if self.offline or not self.rss_feeds:
            return self._load_sample()
        return self._load_live()

    def _load_sample(self) -> pd.DataFrame:
        path = _write_sample_csv("news_sample.csv", _SAMPLE_NEWS)
        df = pd.read_csv(path, parse_dates=["published_date"])
        df = df[df["community"].isin(self.communities)]
        logger.info("NewsLoader: loaded %d headlines from sample CSV", len(df))
        return df.reset_index(drop=True)

    def _load_live(self) -> pd.DataFrame:
        try:
            import feedparser  # noqa: PLC0415

            rows = []
            for community, feeds in self.rss_feeds.items():
                if community.title() not in self.communities:
                    continue
                for url in feeds:
                    feed = feedparser.parse(url)
                    for entry in feed.entries:
                        score = self._vader_score(entry.get("title", ""))
                        rows.append(
                            {
                                "community": community.title(),
                                "headline": entry.get("title", ""),
                                "source": feed.feed.get("title", url),
                                "published_date": entry.get("published", ""),
                                "sentiment_score": score,
                            }
                        )
            df = pd.DataFrame(rows)
            logger.info("NewsLoader: fetched %d live headlines", len(df))
            return df
        except Exception as exc:
            logger.warning("NewsLoader live fetch failed (%s); falling back to sample.", exc)
            return self._load_sample()

    @staticmethod
    def _vader_score(text: str) -> float:
        """Compute a simple positive/negative polarity score."""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: PLC0415
            sia = SentimentIntensityAnalyzer()
            return sia.polarity_scores(text)["compound"]
        except Exception:
            # Fallback: count positive/negative words naively
            positive = {"great", "good", "new", "award", "opens", "record", "best"}
            negative = {"crisis", "crime", "layoff", "delay", "closes", "emergency", "surge"}
            words = set(text.lower().split())
            return (len(words & positive) - len(words & negative)) / max(len(words), 1)


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

class CommunityPulseDataLoader:
    """
    Convenience wrapper that loads all three modalities at once.

    Returns
    -------
    dict with keys: "reddit", "census", "news"
    """

    def __init__(
        self,
        communities: list[str] | None = None,
        offline: bool = True,
        reddit_credentials: dict | None = None,
        census_api_key: str | None = None,
        rss_feeds: dict | None = None,
    ) -> None:
        self.reddit = RedditLoader(
            communities=communities,
            offline=offline,
            reddit_credentials=reddit_credentials,
        )
        self.census = CensusLoader(
            communities=communities,
            offline=offline,
            census_api_key=census_api_key,
        )
        self.news = NewsLoader(
            communities=communities,
            offline=offline,
            rss_feeds=rss_feeds,
        )

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all three modalities and return them in a dict."""
        data = {
            "reddit": self.reddit.load(),
            "census": self.census.load(),
            "news": self.news.load(),
        }
        logger.info(
            "CommunityPulseDataLoader: reddit=%d, census=%d, news=%d",
            len(data["reddit"]),
            len(data["census"]),
            len(data["news"]),
        )
        return data
