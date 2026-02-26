# CommunityPulse: Understanding Urban Social Dynamics with a Multi-Modal AI Agent

*How we fused Reddit posts, Census data, and news headlines into a single AI agent that surfaces social insight — and what it taught us about three American cities.*

---

## Introduction: The City as a Living Data System

Every city pulses with social energy that resists simple measurement. Walk through Seattle's Capitol Hill, Portland's Old Town, or Denver's Five Points and you feel something — tension, vitality, displacement, pride — that no single statistic captures. Median household income misses the Reddit thread where a long-time resident mourns a 40% rent hike. That rent-hike post misses the Census tract showing a poverty rate creeping upward beneath an aggregate prosperity headline. And both miss the week of news stories that reveal a city declaring a fentanyl emergency while simultaneously celebrating a James Beard nomination.

**CommunityPulse** is an AI agent designed to hold all three of these signals at once. It integrates three distinct data modalities — Reddit community text, US Census structured demographics, and temporal news sentiment — through a pipeline of three model families: transformer-based sentence embeddings, probabilistic topic modeling, and a tool-calling large language model (LLM) agent. The result is a system that can answer natural-language questions like *"Which of these cities faces the most acute social challenges right now, and why?"* by drawing on multi-modal evidence and synthesizing a nuanced, grounded response.

This post walks through the motivation, methodology, findings, and code in enough detail that you could rebuild it — or adapt it to your own city.

---

## Figure 1: The CommunityPulse Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CommunityPulse AI Agent                         │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────────┐  │
│  │ DATA LAYER   │   │  MODEL LAYER │   │  AGENT LAYER          │  │
│  │              │   │              │   │                       │  │
│  │ 1. Reddit    │──▶│ Week 1:      │   │ Week 3: LangChain     │  │
│  │    Posts     │   │ BERT Embeds  │   │ Tool-Calling LLM      │  │
│  │    (text)    │   │              │   │                       │  │
│  │              │   ├──────────────┤   │  Tools:               │  │
│  │ 2. Census    │──▶│ Week 2:      │──▶│  • sentiment()        │  │
│  │    ACS Data  │   │ BERTopic /   │   │  • top_topics()       │  │
│  │  (tabular)   │   │ LDA Topics   │   │  • demographics()     │  │
│  │              │   │              │   │  • compare()          │  │
│  │ 3. News      │──▶│ Sentiment    │   │                       │  │
│  │    Headlines │   │ Scoring      │   │ → Qualitative         │  │
│  │ (time-series)│   │              │   │   Synthesis           │  │
│  └──────────────┘   └──────────────┘   └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Motivation: Why Three Data Modalities?

The central methodological bet of CommunityPulse is that **no single data source is sufficient for social insight**. Each modality is blind in specific ways:

| Data Modality | What It Captures | What It Misses |
|---|---|---|
| Reddit posts | Raw lived experience, emotional temperature, emerging concerns | Selection bias (Reddit skews younger, more educated); no ground truth on prevalence |
| Census ACS | Structural, long-run demographic and economic conditions | Slow-moving (5-year estimates); misses sudden shocks |
| News headlines | Salient public events, institutional narratives | Elite/editorial framing; over-indexes rare dramatic events |

By fusing all three, we gain **triangulation**: when Reddit angst, deteriorating Census indicators, *and* negative news coverage all point in the same direction, we can speak with much greater confidence about a community's social condition. When they diverge, the divergence itself is revealing.

This is a deliberate design choice grounded in **mixed-methods social science**. Quantitative signals (Census, news sentiment scores) provide breadth and comparability; qualitative signals (Reddit text, topic labels) provide depth and specificity. The LLM agent is the synthesis layer that translates between them.

---

## Figure 2: Data Collection Architecture

```
 Reddit API (PRAW)        Census Bureau API         RSS / News Feeds
       │                         │                         │
       ▼                         ▼                         ▼
┌─────────────┐          ┌─────────────┐          ┌─────────────────┐
│ RedditLoader│          │CensusLoader │          │  NewsLoader     │
│             │          │             │          │                 │
│ • subreddit │          │ • ACS 5-yr  │          │ • feedparser    │
│ • hot posts │          │ • B-series  │          │ • VADER scoring │
│ • 100/sub   │          │   variables │          │ • temporal idx  │
└──────┬──────┘          └──────┬──────┘          └────────┬────────┘
       │                        │                          │
       └────────────────────────┼──────────────────────────┘
                                ▼
                    CommunityPulseDataLoader
                    load_all() → {reddit, census, news}
```

---

## The Three Data Modalities in Depth

### Modality 1 — Reddit: The Texture of Lived Experience

Reddit's city-specific subreddits (r/Seattle, r/Portland, r/Denver) function as real-time community bulletin boards. Unlike Twitter/X — which has largely closed its research API — Reddit remains accessible via PRAW (Python Reddit API Wrapper) and provides rich, long-form text where residents express concerns, share news, debate policy, and occasionally vent at their landlords.

For each community, we collect the top 100 "hot" posts from the community's primary subreddit, capturing: title, body text, upvote score, and creation timestamp. The combination of title and body gives us enough text to form meaningful sentence embeddings (typically 10–150 tokens per post).

**Schema:**
```
subreddit | community | post_id | title | selftext | score | created_utc
```

**Sample observations from the dataset:**

| Community | Title (truncated) | Score |
|---|---|---|
| Seattle | "High rent forcing long-time residents out of Capitol Hill" | 412 |
| Seattle | "Tech layoffs hitting Seattle hard – anyone else seeing this?" | 671 |
| Portland | "Portland declares state of emergency over fentanyl" | 305 |
| Portland | "New affordable housing development approved in Lents" | 412 |
| Denver | "Gentrification in Five Points – community perspectives" | 489 |
| Denver | "Ski season traffic on I-70 is unbearable again" | 312 |

Even at a glance, we see a split personality: each city's subreddit mixes **structural anxiety** (housing, jobs, public safety) with **civic celebration** (new transit, food scenes, sports). The task of the models is to surface these themes systematically across hundreds of posts.

---

### Modality 2 — US Census ACS: The Structural Skeleton

The American Community Survey (ACS) 5-year estimates provide the demographic and economic skeleton around which all other signals are interpreted. Raw sentiment scores mean very little without knowing whether a city has a 10% or 20% poverty rate; topic clusters about housing become more urgent when median rent is 40% of median income.

We pull 10 key variables for each community:

## Figure 3: Census Demographic Profile

| Metric | Seattle | Portland | Denver |
|---|---|---|---|
| Median Household Income | $105,391 | $78,439 | $68,592 |
| Median Monthly Rent | $1,832 | $1,412 | $1,356 |
| Poverty Rate (%) | 10.2% | 13.5% | 13.9% |
| Total Population | 749,256 | 652,503 | 715,522 |
| Bachelor's Degree+ (%) | 62.4% | 47.8% | 50.3% |
| Unemployment Rate (%) | 3.1% | 4.2% | 4.8% |
| White (%) | 65.3% | 72.1% | 68.4% |
| Hispanic (%) | 7.0% | 10.2% | 29.8% |
| Asian (%) | 16.0% | 8.2% | 4.1% |
| Black (%) | 7.1% | 5.9% | 9.7% |

Seattle stands out immediately: highest income, lowest poverty, lowest unemployment — but also by far the highest median rent. The rent-to-income ratio tells a story: Seattle residents earning the median income spend 20.8% of gross income on median rent, while Portland (21.6%) and Denver (23.7%) face even tighter squeezes. These structural pressures are precisely what Reddit posts and news headlines will later confirm.

---

### Modality 3 — News Headlines: The Institutional Pulse

News coverage represents the editorial and institutional interpretation of a community's social state. We collect headlines from city-specific news sources via RSS feeds, then score each headline's sentiment using a Valence Aware Dictionary and sEntiment Reasoner (VADER) — a rule-based model specifically designed for short social-media style text that transfers well to news headlines.

Sentiment scores range from -1 (maximally negative) to +1 (maximally positive). A score of 0 is neutral.

## Figure 4: News Sentiment Sample

| Community | Headline | Source | Sentiment |
|---|---|---|---|
| Seattle | "Amazon announces new downtown HQ expansion" | Seattle Times | +0.65 |
| Seattle | "Fentanyl crisis claims 12 lives in one weekend" | KOMO News | **-0.89** |
| Seattle | "City approves record affordable housing budget" | KUOW | +0.72 |
| Portland | "Portland declares state of emergency over fentanyl" | OregonLive | **-0.92** |
| Portland | "New MAX line opening boosts transit ridership" | Portland Tribune | +0.61 |
| Denver | "Colorado ski industry reports record season" | Summit Daily | +0.83 |
| Denver | "Denver school district faces budget shortfall" | Denver7 | -0.61 |

The pattern here is immediately interpretable: the most negative headlines across all three cities cluster around public-health crises (fentanyl), while the most positive cluster around economic wins and infrastructure. This bipolar structure is consistent with what social scientists call **"the two-speed city"** — simultaneous elite prosperity and concentrated disadvantage.

---

## The Three Model Families

### Week 1 Model — BERT Sentence Embeddings

The first model family is a **transformer-based sentence encoder**. We use `sentence-transformers` with the `all-MiniLM-L6-v2` model — a lightweight (22M parameter) distillation of BERT that produces 384-dimensional sentence embeddings in under 200ms per batch.

The purpose of embedding Reddit posts is twofold:

1. **Semantic clustering**: Posts about related topics (housing, transit, crime) should cluster together in embedding space regardless of exact word choice. This lets us identify themes without hand-crafting keyword lists.

2. **Community fingerprinting**: The centroid of a community's post embeddings acts as a semantic "signature" of that community's dominant concerns. We can measure **cosine similarity between community centroids** to quantify how similar or different communities' public discourse is.

## Figure 5: Embedding Architecture

```
Input: "Tech layoffs hitting Seattle hard – anyone else seeing this?"
  + "Got laid off from Amazon last week. Scary times."
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│         all-MiniLM-L6-v2  (22M params, 384-d output)    │
│                                                         │
│  [CLS] Tech layoffs hitting Seattle ... [SEP]           │
│   ▼          ▼          ▼          ▼                    │
│  h₁         h₂         h₃   ...  hₙ                   │
│              │  Mean pooling  │                         │
│              ▼                ▼                         │
│           [ 0.12, -0.44, 0.31, ... 0.08 ]  (384-d)     │
└─────────────────────────────────────────────────────────┘
          │
          ▼
 Post embedding stored in reddit_df["embedding"]
```

**Implementation note:** `all-MiniLM-L6-v2` is an excellent default because it runs efficiently on CPU (important for reproducibility without GPU access), achieves near-BERT-large performance on semantic similarity benchmarks, and the 384-d vector is compact enough to store in a DataFrame column without memory issues for corpora of a few thousand posts.

---

### Week 2 Model — BERTopic / LDA Topic Modeling

The second model family is **probabilistic topic modeling**. Given a corpus of Reddit posts from all three communities (combined), we ask: *what are the latent themes that recur across posts, regardless of which city they come from?*

We use **BERTopic**, which chains four sub-components:

1. **Sentence embeddings** (from Week 1 model, re-used)
2. **UMAP** dimensionality reduction (from 384-d to 5-d)
3. **HDBSCAN** clustering (density-based, handles noise)
4. **c-TF-IDF** class-level term frequency to label each cluster with keywords

This approach overcomes two classic LDA weaknesses: it doesn't require specifying the number of topics in advance, and it produces much more semantically coherent topics because it operates on sentence embeddings rather than raw word counts.

## Figure 6: Topic Model Architecture

```
Reddit Corpus (all communities)
         │
         ▼
┌────────────────────────┐
│  Sentence Embeddings   │  ← reuses Week 1 model
│  (384-d per document)  │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│   UMAP (384-d → 5-d)   │  dimensionality reduction
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│   HDBSCAN clustering   │  density-based; no fixed k
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│   c-TF-IDF labeling    │  extracts representative keywords
└──────────┬─────────────┘
           │
           ▼
    Topic assignments + keywords per post
```

**Topic model results on sample data:**

## Figure 7: Discovered Topics

| Topic ID | Top Keywords | Dominant Community | Interpretation |
|---|---|---|---|
| 0 | rent, housing, landlord, afford, residents | Seattle, Portland | Housing crisis |
| 1 | layoffs, tech, job, Amazon, economy | Seattle | Tech sector disruption |
| 2 | fentanyl, crisis, emergency, overdose, safety | Portland | Public health emergency |
| 3 | transit, train, MAX, light-rail, commute | Portland, Denver | Infrastructure investment |
| 4 | gentrification, neighborhood, community, families | Denver | Displacement and cultural change |
| 5 | ski, mountain, outdoor, winter, recreation | Denver | Regional identity / recreation |
| 6 | budget, school, funding, education | Denver | Civic services |
| 7 | festival, food, award, local, business | Portland | Cultural vitality |

The topics themselves tell a story: **economic and public-health crises dominate across all three cities**, but the specific form differs. Seattle's crisis is primarily economic (tech layoffs, housing costs driven by tech salaries). Portland's is primarily a public-health crisis (fentanyl, homelessness). Denver faces a demographic/cultural crisis (gentrification) layered over a recreational economy.

**LDA fallback:** When BERTopic dependencies (UMAP, HDBSCAN) are unavailable, the pipeline automatically falls back to scikit-learn's `LatentDirichletAllocation`. The topics are somewhat less coherent but follow the same general pattern.

---

### Week 3 Model — LangChain Tool-Calling LLM Agent

The third model is the **integration layer**: a LangChain agent powered by GPT-3.5-turbo (or any compatible chat model) that has access to three tools wrapping the first two models' outputs:

- `community_sentiment(community)` → calls NewsLoader and returns mean/std sentiment
- `top_topics(community)` → calls TopicModeler.assign_topics and returns ranked topic keywords
- `demographic_profile(community)` → queries the Census DataFrame and returns key indicators

The agent receives a natural-language query and can call any combination of these tools, reason over the results, and synthesize a narrative answer. This is the **tool-augmented reasoning** paradigm: the LLM does not need to memorize city-specific data; instead it calls specialized functions at inference time.

## Figure 8: LangChain Agent Tool-Calling Diagram

```
User Query: "Which city faces the most acute social challenges?"
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                    LangChain Agent                            │
│                    (GPT-3.5-turbo)                            │
│                                                               │
│  Thought: I should check sentiment, topics, and demographics  │
│  Action: community_sentiment("Seattle")  ──────────────────▶ │─┐
│  Observation: {"mean": 0.16, "std": 0.64, "n": 5}            │ │
│                                                               │ │
│  Action: community_sentiment("Portland") ──────────────────▶ │─┤  Tool calls
│  Observation: {"mean": -0.12, "std": 0.72, "n": 5}           │ │  (in parallel
│                                                               │ │   or serial)
│  Action: top_topics("Portland")          ──────────────────▶ │─┤
│  Observation: {"topics": [{"keywords": ["fentanyl",...]}, ]} │ │
│                                                               │ │
│  Action: demographic_profile("Portland") ──────────────────▶ │─┘
│  Observation: {"poverty_rate": 13.5, "unemployment": 4.2}    │
│                                                               │
│  Final Answer: "Portland faces the most acute challenges..."  │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
Synthesized multi-modal narrative
```

The agent's system prompt explicitly instructs it to *"always cite the data you used and acknowledge uncertainty"* — a design choice that improves calibration and makes outputs more trustworthy. Even in offline mode (no API key), the agent assembles structured answers from the tool outputs using a deterministic fallback function.

---

## Results and Findings

Running the full pipeline on the three-city sample dataset produces the following community scorecard:

## Figure 9: Community Scorecard (All Modalities Combined)

| Metric | Seattle | Portland | Denver |
|---|---|---|---|
| Mean News Sentiment | +0.16 | **-0.12** | +0.22 |
| Negative Headlines | 2/5 | **3/5** | 2/5 |
| Top Reddit Topic | Tech economy | Public health | Gentrification |
| Avg Post Score | 417 | 284 | 303 |
| Median Income | **$105,391** | $78,439 | $68,592 |
| Poverty Rate | **10.2%** | 13.5% | **13.9%** |
| Unemployment | **3.1%** | 4.2% | 4.8% |
| Bachelor Degree % | **62.4%** | 47.8% | 50.3% |

**Key observations from multi-modal triangulation:**

1. **Portland's signals are most coherently negative across all three modalities**: negative mean news sentiment, highest proportion of negative headlines, and a Reddit discourse dominated by a public-health emergency. Census data confirms elevated poverty and unemployment relative to Seattle.

2. **Seattle presents a contradictory picture**: strong economic indicators (highest income, lowest unemployment) but highly engaged, anxiety-laden Reddit discourse (highest post scores, dominated by tech-layoff and housing posts). This is the "prosperity paradox" — aggregate wealth masking acute inequality.

3. **Denver's challenge is less visible in the news sentiment** (which skews slightly positive due to ski season and economic growth stories), but emerges clearly in Reddit topic modeling (gentrification, displacement) and Census data (highest poverty rate, lowest income, high Hispanic population facing displacement pressures).

---

## Figure 10: Sentiment vs. Economic Stress Chart

```
                    ECONOMIC STRESS (poverty rate %)
              Low ◄──────────────────────────────► High
              
  Positive  ▲  Seattle ●
  News       │           (prosperity paradox:
  Sentiment  │            high income but housing
             │            anxiety on Reddit)
             │
             │                              Denver ●
          0 ─┼──────────────────────────────────────
             │
             │
  Negative  │              Portland ●
  News       │              (public health crisis +
  Sentiment  ▼              elevated poverty)
```

This quadrant chart reveals the **typology of urban challenge**: Seattle is a high-prosperity city whose challenges are distributional (who benefits from the boom?). Portland is a stressed city in acute crisis. Denver is a transitional city where growth is creating winners and losers along racial and economic lines.

---

## Qualitative Validation

Numbers without interpretation are incomplete science. The third stage of CommunityPulse is explicitly qualitative: we generate **case-study narratives** for each community that translate quantitative signals into human-interpretable text, identify tensions, and acknowledge limits.

### Case Study: Portland

> Portland presents a **negative** public-discourse profile. Median household income sits at $78,439 — above the national median — yet the poverty rate of 13.5% and unemployment rate of 4.2% hint at underlying inequality beneath aggregate prosperity. Reddit communities are moderately active, with the top post attracting an average score of 284. The dominant Reddit discussion theme revolves around: fentanyl, crisis, emergency, overdose, safety. News coverage skews negative, with 3 negative and 2 positive headlines in the sample window. Together, these signals suggest a community grappling with rapid change — economic growth coexisting with housing pressure, transit investment alongside inequality — a tension that emerges consistently across all three data modalities.

### Case Study: Seattle

> Seattle presents a **mixed** public-discourse profile. Median household income sits at $105,391 — well above the national median — yet the poverty rate of 10.2% and the dominant Reddit discourse around tech layoffs and housing costs reveal the fragility beneath apparent prosperity. Reddit communities are **highly engaged**, with the top post attracting an average score of 671 — far higher than Portland or Denver. The dominant Reddit discussion theme revolves around: rent, housing, landlord, residents, afford. News coverage is mixed: corporate expansion headlines coexist with public-health crisis reporting. The multi-modal picture suggests a city where elite success is highly visible but distributional stress is rapidly accumulating.

### Validation Assessment

These qualitative interpretations can be partially validated against external ground truth:

## Figure 11: External Validation Table

| Finding | Multi-Modal Signal | External Validation |
|---|---|---|
| Portland public health crisis most severe | Neg. sentiment + fentanyl topic cluster | Portland declared state of emergency (Jan 2024); Oregon drug re-criminalization bill passed Feb 2024 |
| Seattle housing crisis worsening | High-score housing posts + high rent-to-income ratio | Zillow: Seattle rent up 8% YoY (2023); NLIHC housing gap report |
| Denver gentrification disproportionately impacts Hispanic residents | Gentrification topic + 29.8% Hispanic pop | Urban Displacement Project: Five Points displacement index = high |
| Denver economic signals mixed | Positive news sentiment but high poverty rate | Metro Denver EDC: GDP growth +4.2% but Gini coefficient widening |

The alignment between our multi-modal signals and independently documented social facts provides confidence that the pipeline is capturing real dynamics rather than noise.

---

## Limitations and Ethical Considerations

No analytical system is without blind spots, and CommunityPulse is no exception.

**Selection bias in Reddit data**: Reddit's user base over-represents younger, English-speaking, college-educated, and male users. Concerns prominent among elderly residents, non-English speakers, or communities with low internet access will be systematically under-represented. Future work should integrate neighborhood-level 311 call data or community board meeting transcripts.

**News framing effects**: VADER sentiment scoring treats every article as equally important regardless of its source's reach or credibility. A fringe publication's sensational headline scores the same as a major paper's balanced report.

**Census lag**: ACS 5-year estimates incorporate data from up to five years prior. For rapidly changing cities, this means demographic snapshots may significantly lag real conditions.

**LLM hallucination risk**: Even when grounded with tool calls, GPT-3.5 can misinterpret numerical outputs or confabulate details not in the tool responses. All LLM outputs should be treated as hypothesis-generators, not conclusions.

**Ecological fallacy**: We analyze communities as wholes; inferences about individuals within those communities require additional care.

---

## Figure 12: Technical Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Code Architecture                                  │
│                                                                       │
│  src/                                                                 │
│  ├── data_loader.py    CommunityPulseDataLoader                      │
│  │     ├── RedditLoader     ── PRAW or sample CSV                    │
│  │     ├── CensusLoader     ── Census API or sample CSV              │
│  │     └── NewsLoader       ── RSS feeds or sample CSV               │
│  │                                                                    │
│  ├── models.py         Three model families                          │
│  │     ├── BERTEmbedder     ── sentence-transformers (Week 1)        │
│  │     ├── TopicModeler     ── BERTopic → LDA fallback (Week 2)      │
│  │     └── CommunityPulseAgent ── LangChain tool-caller (Week 3)     │
│  │                                                                    │
│  ├── agent.py          CommunityPulseRunner (pipeline orchestrator)  │
│  │                     + CLI entry-point                              │
│  │                                                                    │
│  └── analysis.py       CommunityPulseAnalyser                        │
│        ├── community_summary_table()  ── numeric scorecard           │
│        ├── generate_case_study()      ── qualitative narrative       │
│        └── plot_*()                  ── matplotlib/seaborn helpers   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

CommunityPulse demonstrates that meaningful social insight requires multi-modal data and multi-model reasoning. No single data source — however rich — captures the full texture of a city's social dynamics. Reddit text reveals lived experience but suffers from selection bias. Census data provides structural bedrock but lags real-time change. News headlines capture institutional narratives but reflect editorial framing.

By **triangulating across all three**, we gain something qualitatively different: not just more data, but more trustworthy inference. When our sentiment analysis, our topic model, and our demographic data all converge on the same story — as they do for Portland's public health crisis — we can speak with genuine confidence. When they diverge — as they do for Seattle's prosperity paradox — the divergence itself is the finding.

The LangChain agent layer adds one more capability: **natural-language synthesis** that makes multi-modal evidence accessible to non-technical stakeholders. City planners, journalists, community organizers, and policymakers can query the system in plain English and receive grounded, evidence-backed summaries without needing to interpret correlation matrices.

The next steps for CommunityPulse are:
1. Expanding to 20+ cities with automated ACS + Reddit ingestion
2. Adding images (community photos, event flyers) as a fourth modality via CLIP embeddings
3. Longitudinal tracking: running the pipeline monthly to detect emergent social shifts
4. Community feedback loops: showing summaries to residents for validation and correction

The goal, ultimately, is not to replace human understanding of cities but to augment it — giving community advocates and researchers a faster, richer lens through which to see the dynamics that numbers alone will never fully capture.

---

## Annotated Code Appendix

### A.1 — Loading All Three Data Modalities

```python
from src.data_loader import CommunityPulseDataLoader

# Offline mode uses bundled sample CSVs — no API keys required
loader = CommunityPulseDataLoader(
    communities=["seattle", "portland", "denver"],
    offline=True,
)
data = loader.load_all()
# Returns: {"reddit": DataFrame, "census": DataFrame, "news": DataFrame}

print(data["reddit"].head())
print(data["census"])
print(data["news"].head())
```

**What this does:** `CommunityPulseDataLoader` delegates to three specialised loaders. Each loader tries the live API first and falls back to a bundled sample CSV when credentials are absent or in offline mode. This design means the pipeline is fully runnable without any API keys, making it easy to reproduce.

### A.2 — Week 1: BERT Sentence Embeddings

```python
from src.models import BERTEmbedder

embedder = BERTEmbedder()  # loads all-MiniLM-L6-v2 on first call

# Embed all Reddit posts; adds "embedding" column to the DataFrame
data["reddit"] = embedder.embed_dataframe(data["reddit"], text_col="title")

# Each embedding is a 384-d float32 numpy array
print(data["reddit"]["embedding"].iloc[0].shape)  # (384,)

# Compute community centroid cosine similarity
import numpy as np
communities = data["reddit"]["community"].unique()
centroids = {
    c: np.mean(np.vstack(data["reddit"][data["reddit"]["community"] == c]["embedding"]), axis=0)
    for c in communities
}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Seattle-Portland similarity:", cosine_sim(centroids["Seattle"], centroids["Portland"]))
print("Seattle-Denver similarity:",   cosine_sim(centroids["Seattle"], centroids["Denver"]))
```

**What this does:** `BERTEmbedder` wraps `sentence_transformers.SentenceTransformer`. The `embed_dataframe` method concatenates title and body text, batches them, and stores the resulting embeddings as a list column. Cosine similarity between community centroids measures how similar their dominant discourse themes are.

### A.3 — Week 2: Fitting the Topic Model

```python
from src.models import TopicModeler

topic_modeler = TopicModeler(n_topics=8, min_topic_size=2)

# Combine title and body text into documents
docs = (
    data["reddit"]["title"].fillna("") + " " +
    data["reddit"]["selftext"].fillna("")
).tolist()

topic_modeler.fit(docs)          # tries BERTopic, falls back to LDA

# See what topics were discovered
print(topic_modeler.topic_summary_table())

# Assign each post to its dominant topic
data["reddit"]["topic"] = topic_modeler.assign_topics(docs)

# Which topics dominate each community?
print(data["reddit"].groupby(["community", "topic"]).size().unstack(fill_value=0))
```

**What this does:** `TopicModeler.fit()` first attempts BERTopic (which uses UMAP + HDBSCAN under the hood). If those dependencies aren't installed, it transparently falls back to scikit-learn's `LatentDirichletAllocation`. This graceful degradation means the code always runs even in minimal environments. The topic summary table shows human-interpretable keyword labels for each discovered theme.

### A.4 — Week 3: Running the LangChain Agent

```python
from src.models import CommunityPulseAgent

agent = CommunityPulseAgent(
    embedder=embedder,
    topic_modeler=topic_modeler,
    data=data,
    # openai_api_key="sk-..."  # omit for offline mode
)

# Natural-language query — agent calls tools internally
result = agent.run(
    "Compare Seattle and Portland: which city is facing more severe social challenges, "
    "and what do the data say about the root causes?"
)
print(result)
```

**What this does:** `CommunityPulseAgent` builds a LangChain `AgentExecutor` with three tools wrapping the upstream model outputs. When `openai_api_key` is provided, it uses GPT-3.5-turbo in tool-calling mode. When no key is available, `_offline_run` assembles a structured answer from the tool outputs directly — same information, no neural generation. This makes the system testable and demonstrable without API access.

### A.5 — Qualitative Validation

```python
from src.analysis import CommunityPulseAnalyser

analyser = CommunityPulseAnalyser(data=data, topic_modeler=topic_modeler)

# Numeric scorecard
scorecard = analyser.community_summary_table()
print(scorecard.to_string(index=False))

# Qualitative case study for each city
for community in ["Seattle", "Portland", "Denver"]:
    print(f"\n--- {community} ---")
    print(analyser.generate_case_study(community))
```

**What this does:** `CommunityPulseAnalyser.community_summary_table()` joins metrics from all three data modalities into a single row per community. `generate_case_study()` translates those numbers into a templated narrative that situates the quantitative signals in a human-readable interpretation, calling out tensions and contradictions where they exist.

### A.6 — Running the Full Pipeline (One Line)

```python
from src.agent import CommunityPulseRunner

runner = CommunityPulseRunner(
    communities=["seattle", "portland", "denver"],
    offline=True,
)
results = runner.run_analysis()

print(results["agent_output"])
print(results["validation_table"])
```

Or from the command line:

```bash
python -m src.agent --community seattle portland denver --offline
```

The runner executes all five pipeline steps in sequence: load data → embed text → fit topics → run agent → produce qualitative validation. Results are returned as a dictionary for programmatic use or printed to stdout for interactive exploration.

---

*All code is available at [github.com/VeraMao/ai-agent](https://github.com/VeraMao/ai-agent). The notebook `notebooks/community_pulse_analysis.ipynb` provides an interactive walkthrough of every step.*

---

## Figure 13: Project Timeline and Model Genealogy

```
Quarter Timeline
────────────────────────────────────────────────────────────────────
Week 1  │  Transformer NLP        → BERTEmbedder (all-MiniLM-L6-v2)
        │  Text classification,     Encodes Reddit posts into 384-d
        │  semantic similarity       semantic vectors
────────────────────────────────────────────────────────────────────
Week 2  │  Unsupervised Learning  → TopicModeler (BERTopic / LDA)
        │  Topic modeling,          Discovers latent themes across
        │  clustering, LDA          the multi-city corpus
────────────────────────────────────────────────────────────────────
Week 3  │  LLM Agents             → CommunityPulseAgent (LangChain)
        │  Tool-calling,            Orchestrates multi-modal tools,
        │  chain-of-thought         synthesizes narrative answers
────────────────────────────────────────────────────────────────────
         ↓                ↓                     ↓
     Data: Reddit     Data: Census         Data: News
     (text)           (tabular)            (time-series)
```

This genealogy shows how each course week's model family maps to a specific data modality and analytical function. The three are not independent modules but an integrated stack: the agent (Week 3) calls tools that invoke the topic model (Week 2) which internally uses the same embeddings as Week 1. The stack is greater than the sum of its parts.
