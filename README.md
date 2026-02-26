# CommunityPulse AI Agent

**Urban Social Dynamics Through Multi-Modal AI Analysis**

An AI agent that fuses three distinct data modalities — Reddit community text, U.S. Census demographic tables, and temporal news sentiment — to surface social insights about urban communities. The agent integrates three model families (transformer NLP, probabilistic topic modeling, and a tool-calling LLM agent) into a single, unified analysis pipeline.

## Project Overview

| Requirement | Implementation |
|---|---|
| ≥ 3 data modalities | Reddit posts (text), Census ACS data (tabular), news headlines (time-series) |
| ≥ 3 model types | BERT sentence embeddings, BERTopic/LDA topic model, LangChain LLM agent |
| Qualitative validation | Per-community case studies with human-interpretable summaries |

## Repository Structure

```
ai-agent/
├── blog_post.md              ← Public-facing blog post (3600+ words, 13+ figures)
├── requirements.txt          ← Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py        ← Loaders for all three data modalities
│   ├── models.py             ← BERT embeddings + BERTopic + LangChain agent setup
│   ├── agent.py              ← Main multi-modal AI agent
│   └── analysis.py           ← Analysis helpers and qualitative validation
├── notebooks/
│   └── community_pulse_analysis.ipynb  ← End-to-end walkthrough notebook
└── data/
    └── sample/               ← Small sample data files for offline testing
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the agent on the sample data (no API keys required for offline mode)
python -m src.agent --community seattle --offline

# Or open the notebook
jupyter notebook notebooks/community_pulse_analysis.ipynb
```

## Blog Post

The full write-up — including motivation, methodology, results, and an annotated code appendix — is available in [`blog_post.md`](blog_post.md).

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list
- Optional: `OPENAI_API_KEY` for the LLM agent; `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` for live Reddit data
