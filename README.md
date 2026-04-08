# Encoding Emotion in Music: A Multimodal AI Benchmark Across Audio, Lyrics, and Language Models  
*A computational social science exploration of how machines infer musical emotion from sound, text, and human-generated tags.*

---

## Description

Music is one of the most emotionally expressive forms of human communication, but computationally identifying that emotional meaning is difficult because emotion in music does not live in a single place. It can be expressed through **acoustic structure**, **lyrical content**, and **listener interpretation**, all of which may align or diverge.

This project builds a **multimodal benchmark for musical emotion classification** by comparing three different approaches:

- **Audio-based machine learning models** trained on Spotify acoustic features  
- **Lyric-based transformer classification** using Genius lyrics  
- **LLM-based reasoning** from structured acoustic feature prompts alone  

To create labels at scale, we use a **weak supervision pipeline** based on **Last.fm user tags**, which are mapped into six core emotions:

- joy  
- sadness  
- anger  
- fear  
- surprise  
- disgust  

By comparing these modalities, the project asks a broader computational social science question: **how do machines infer feeling from cultural data, and what changes depending on the signal they receive?**

---

## Research Question

**To what extent do acoustic features and lyrics yield consistent or divergent emotion predictions in popular music?**

More specifically, this project asks:

1. Can **Spotify acoustic features alone** reliably predict musical emotion?
2. Do **lyrics** express the same emotional signals that listeners assign through tags?
3. Can a **large language model** infer emotion from **numeric acoustic descriptors alone**, without hearing the song or reading the lyrics?

---

## Data Sources

This project combines three major data sources:

### 1. Last.fm User Tags
Crowd-generated tags are used as a proxy for listener-perceived emotion.  
These tags are converted into structured emotion labels using a transformer-based weak supervision pipeline.

### 2. Spotify Acoustic Features
Quantitative features such as:

- danceability  
- energy  
- valence  
- tempo  
- loudness  
- acousticness  
- instrumentalness  
- speechiness  
- liveness  

These features are used for supervised multilabel emotion prediction.

### 3. Genius Lyrics
Lyrics provide the textual and semantic content of songs, allowing us to compare **artist expression** with **listener interpretation**.

---

## Project Components

The repository contains work spanning several stages of the project:

- **Data preprocessing and dataset construction**
- **Weak supervision for emotion label generation**
- **Baseline machine learning models on acoustic features**
- **Hyperparameter tuning using Dask and Spark**
- **Lyric-based transformer classification**
- **LLM feature-only inference experiment**
- **Model evaluation and results comparison**

---

## File Structure

```text
Emotion-Music-ML
│
├── DataSets/
│   ├── Final_Datset.zip
│   └── rawdataset.txt
│
├── DataCollection&Pre/
│
├── ML_Progress/
│
├── HPTuning_Dask/
│
├── HPTuning_Spark/
│
├── Initial_running/
│   ├── requirement.txt
│   └── spotify_secret.py
│
├── Lyris_Genius/
│
├── Results/
│
├── Data_Preprocessing.ipynb
├── ML_Progress.ipynb
├── LLM_featureOnly_experiment.ipynb
│
├── spotify_secret.py
└── README.md
```
## Folder Descriptions

### DataSets/

Contains both the **final processed dataset used in experiments** and references to the **raw datasets**.

- `Final_Datset.zip` — Final dataset combining Spotify acoustic features with emotion labels  
- `rawdataset.txt` — Links and descriptions for downloading raw datasets (Spotify, Last.fm)

---

### DataCollection&Pre/

Scripts and workflows used for **data collection and preprocessing**, including dataset cleaning, merging, and preparation.

---

### ML_Progress/

Code used for **training and evaluating machine learning models** on acoustic features.

---

### HPTuning_Dask/

Hyperparameter tuning experiments using **Dask for parallel computation**.

---

### HPTuning_Spark/

Hyperparameter tuning experiments using **Spark / EMR clusters**.

---

### Initial_running/

Initial project setup files.

- `requirement.txt` — Python dependencies required to run the project  
- `spotify_secret.py` — Local Spotify API credential configuration

---

### Lyris_Genius/

Pipeline for **lyrics collection and lyric-based emotion classification** using transformer models.

---

### Results/

Contains **model outputs, evaluation summaries, and experiment results**.

---

## Key Notebooks

### Data_Preprocessing.ipynb

Primary notebook used to:

- prepare datasets  
- clean and merge data sources  
- construct the final modeling dataset  

---

### ML_Progress.ipynb

Main notebook for **training and evaluating machine learning models**.

Models include:

- Random Forest  
- K-Nearest Neighbors  
- Multi-Layer Perceptron  

---

### LLM_featureOnly_experiment.ipynb

Experiment testing whether a **large language model can infer emotion from structured Spotify acoustic features alone**.

The model receives only numeric acoustic descriptors and predicts possible emotion labels.

---

### spotify_secret.py

Configuration file for **Spotify API credentials**.  

---

## Methodology Overview

### 1. Weak Supervision Pipeline

To generate emotion labels at scale, Last.fm tags are passed through the transformer model:

`j-hartmann/emotion-english-distilroberta-base`

Only predictions with confidence above **0.8** are retained, which improves precision while reducing coverage. This produces a weakly supervised multilabel dataset for downstream modeling.

### 2. Audio-Based Emotion Prediction

Using Spotify acoustic features, we train three multilabel classifiers:

- Random Forest
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)

These models predict the six core emotions from numeric song features alone.

### 3. Hyperparameter Tuning

We compare:

- Grid Search
- Random Search

across two compute environments:

- Dask
- Spark / EMR

This allows us to compare predictive performance against runtime and infrastructure cost.

### 4. Lyrics Emotion Classification

Lyrics scraped from Genius are processed and passed through a BERT-based emotion classifier:

`bhadresh-savani/bert-base-go-emotion`

Predictions are mapped into the same six-emotion taxonomy for comparison with tag-based labels.

### 5. LLM Feature-Only Experiment

A large language model receives only structured Spotify acoustic features and is asked to infer likely emotions for each song. This experiment tests whether an LLM can reason about affect from abstract numeric descriptors without access to lyrics or audio.

---

## Results

### Audio Model Results

Acoustic-feature-based models performed the strongest overall in the project.

**Random Forest** achieved the best overall performance:

- **Micro-F1:** 0.6766
- **Exact Match Accuracy:** 36.67%
- **Hamming Loss:** 0.2261
- **Macro-F1:** 0.5379
- **Weighted F1:** 0.6547

**MLP** performed very similarly:

- **Micro-F1:** 0.6757
- Slightly lower Macro-F1 and Weighted F1 than Random Forest
- Faster and more computationally efficient

**KNN** performed noticeably worse:

- **Micro-F1:** 0.6332
- **Exact Match Accuracy:** 30.30%
- **Hamming Loss:** 0.2565

These results suggest that acoustic structure is strongly informative for perceived musical emotion, especially when modeled with nonlinear classifiers.

### Hyperparameter Tuning Results

Tuning experiments showed two clear patterns:

- Random Forest generally produced the strongest predictive performance
- Random Search often matched Grid Search performance with much lower runtime
- MLP was much faster to train, making it useful for rapid experimentation
- Larger Spark cluster sizes reduced runtime substantially without changing performance much

Overall, tuning improved efficiency and helped clarify tradeoffs between model quality and compute cost.

### Lyric-Based Transformer Results

The lyric-based classifier achieved an overall accuracy of approximately:

- **Accuracy ≈ 18%**

This is much lower than the audio-based models, but the result is still meaningful. It suggests that lyrics and listener-perceived emotion are not the same thing.

Key findings:

- Anger and joy were easier to detect from lyrics
- Fear, surprise, and disgust were much harder to identify
- Predictions often clustered around a small set of dominant emotions
- Lyrics capture narrative or semantic emotion, but not always the full affective experience of listening

### LLM Feature-Only Results

The LLM experiment showed that a language model can infer some emotional patterns from structured acoustic features alone, but performance varied by category.

Main patterns:

- Joy was the easiest emotion to detect
- Sadness and fear showed moderate performance
- Anger and surprise were harder
- Disgust was rarely predicted correctly

Common error types included:

- Label omission in multilabel cases
- Overreliance on broad signals like high energy or high valence
- Difficulty with emotionally mixed or ambiguous songs

### Overall Interpretation

Across modalities, the results suggest:

- Audio features are the strongest standalone signal for listener-perceived emotion
- Lyrics capture emotional language, but often reflect a different layer of meaning
- LLMs can reason from structured features, but their predictions are less reliable than supervised models
- Weak supervision from human tags remains central for building scalable emotion datasets

---

## Key Takeaways

- Emotion in music is fundamentally multimodal
- Acoustic features, lyrics, and human tags capture different emotional layers
- Weak supervision enables scalable label construction, but the **0.8 threshold** introduces a precision–coverage tradeoff
- Lyrics alone do not fully capture how music feels to listeners
- Future work should move toward multimodal neural models that combine:
  - acoustic features
  - lyric embeddings
  - listener tag distributions

---

## Setup Notes

Some parts of this repository require external credentials or downloaded datasets.

### Spotify Credentials

The file `spotify_secret.py` is used for Spotify API access.  
You will need to create your own credentials and configure this locally.

### Requirements

Install dependencies from the requirements file inside `Initial_running/`:

```bash
pip install -r Initial_running/requirement.txt
```
## Raw Datasets

*(See `Datasets/rawdataset.txt` for more detailed descriptions.)*

Due to size restrictions, raw data files are **not directly uploaded to this repository**. However, all source datasets are publicly available and can be downloaded from the following locations:

- **Spotify Audio Features (8M Tracks)**  
  [Kaggle: 8M Spotify Tracks – Genre & Audio Features](https://www.kaggle.com/datasets/maltegrosse/8-m-spotify-tracks-genre-audio-features)

- **Last.fm User Tags**  
  [Last.fm Tag Annotations (Million Song Dataset)](http://millionsongdataset.com/lastfm/)

- **Tags Database (SQLite Format)**  
  [lastfm_tags.db (LabROSA)](http://labrosa.ee.columbia.edu/~dpwe/tmp/lastfm_tags.db)

- **Track Metadata (Subset)**  
  [lastfm_subset.zip (Million Song Dataset)](http://millionsongdataset.com/sites/default/files/lastfm/lastfm_subset.zip)

All code is provided in this repository. Raw data must be downloaded from public sources (see above).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
