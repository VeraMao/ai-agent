# Encoding Emotion in Music via Acoustic Features  
*A Weakly Supervised Transformer-Based Study*

## 📘 Description

This project investigates how the acoustic properties of music relate to emotional expression by training machine learning models to predict emotion categories from Spotify audio features. Building on research in music psychology and affective computing \parencite{Huron2015, Perlovsky2010, Yang2024}, the study adopts a **weakly supervised** approach where emotion labels are inferred from Last.fm user-generated tags using a **DistilRoBERTa transformer classifier**.

After filtering predictions by confidence and matching them to Spotify audio features via API, the final dataset includes **82,950 tracks** labeled across six basic emotions: *joy, sadness, anger, fear, disgust,* and *surprise*. The study compares four classifiers—**Random Forest, Logistic Regression, K-Nearest Neighbors**, and **Multi-Layer Perceptron**—using a **dynamic top-k multilabel evaluation** scheme.

Rather than focusing solely on accuracy, this project emphasizes **interpretability**: How do different models leverage features like *valence*, *energy*, *loudness*, and *acousticness*? Using **SHAP values** and **permutation importance**, the analysis reveals distinct feature usage patterns and highlights how classification failures often stem from weak or contradictory signals—not random noise.

Key findings show that while Random Forest achieves the best overall performance, emotions like *fear*, *disgust*, and *surprise* remain challenging due to semantic ambiguity and overlapping acoustic cues. This underscores the importance of **multilabel-aware metrics** and interpretability tools in understanding emotion recognition models.

Overall, this work contributes to music emotion recognition by showing how models interpret audio features, where and why they fail, and how evaluation strategies must account for the inherent complexity of musical affect.


---

## ⚙️ Requirements

This project was developed and tested in Python 3.10. The following libraries are required to run the full pipeline, from data collection to model training and evaluation:

```python
# Core machine learning and evaluation
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
shap==0.45.0

# Visualization
matplotlib==3.8.4
seaborn==0.13.2
tqdm==4.66.2

# Spotify API access
spotipy==2.23.0

# SQLite for metadata extraction
sqlite3 (built-in with Python)

# Authentication
# You will need to create a file called `spotify_secret.py` with:
# SPOTIPY_CLIENT_ID = 'your_client_id'
# SPOTIPY_CLIENT_SECRET = 'your_client_secret'
```

## 📁 File Structure

```text
Emotion-Music-ML
├── DataSets/ # Raw and processed datasets
   ├──Final_Datset.zip # Final dataset containing audio features and labels
   ├── rawdataset.txt # Links to download the raw dataset: Kaggle's "Spotify Audio Features" and Last.fm "Tags", "metadata" dataset
├── DataCollection&Pre # Scripts for data collection and preprocessing
├── ML_Progress # Scripts for model training and evaluation
├── spotify_secret.py # Spotify API credentials (you need to set this up using your own credentials)
│
├── Presentation_QMD/ # Quarto slides
│
├── Results/ # All results from the ML pipeline
│ ├── albation_results.csv # Results from ablation study
│ ├── all_model_feature_importance.csv # Feature importance for all models
│ ├── summary_audio_feature_stats.csv # Results from all models
│
├── Encoding_Emotion_in_Music_Paper_Track_FinalPaper.pdf # Final paper in PDF format
├── Requirements.txt # Python package requirements
└── README.md # Project overview and documentation
```

---

## 📁 Raw Datasets (See Datasets/rawdataset.txt for more detailed descriptions.)

Due to size restrictions, raw data files are **not directly uploaded to this repository**. However, all source datasets are publicly available and can be downloaded from the following locations:

- 🎧 **Spotify Audio Features (8M Tracks)**  
  [Kaggle: 8M Spotify Tracks – Genre & Audio Features](https://www.kaggle.com/datasets/maltegrosse/8-m-spotify-tracks-genre-audio-features)

- 🏷️ **Last.fm User Tags**  
  [Last.fm Tag Annotations (Million Song Dataset)](http://millionsongdataset.com/lastfm/)

- 🧠 **Tags Database (SQLite Format)**  
  [lastfm_tags.db (LabROSA)](http://labrosa.ee.columbia.edu/~dpwe/tmp/lastfm_tags.db)

- 🗂️ **Track Metadata (Subset)**  
  [lastfm_subset.zip (Million Song Dataset)](http://millionsongdataset.com/sites/default/files/lastfm/lastfm_subset.zip)

All code is provided in this repository. Raw data must be downloaded from public sources (see above).

---
## Overleaf Repository
For the LaTeX paper and presentation, you can access the Overleaf repository here:
[Overleaf Repository - Encoding Emotion in Music](https://github.com/VeraMao/Encoding-Emotion-in-Music_Overleaf)

The Overleaf repository contains:
```text
├── Archived/ # Archived files from the Overleaf project
├── Graphics/ # Figures used in the paper
└── Perspective/ # LaTeX files for the Perspective paper
```text

---

**Key Findings:**
1. **Emotion classification is feasible using only audio features**  
   Models like Random Forest can effectively classify six emotions (*anger, disgust, fear, joy, sadness, surprise*) from 12 Spotify-derived features. Best micro F1 score reached ~0.68.

2. **Emotional prediction varies across categories**  
   *Joy* and *sadness* were predicted with high accuracy. *Fear*, *disgust*, and *surprise* were more error-prone, due to low salience and higher semantic ambiguity.

3. **Weakly supervised labels are noisy but scalable**  
   Labels were generated by applying a DistilRoBERTa classifier to Last.fm tags. This introduces some semantic drift—especially for genre or aesthetic tags—but enables large-scale annotation.

4. **Interpretability highlights fuzzy boundaries**  
   SHAP analysis shows high-impact features (*valence*, *energy*, *acousticness*) drive predictions. Misclassifications often lacked clear signal, pointing to inherent emotional ambiguity in music.

5. **Ablation studies confirm feature importance**  
   Removing top-ranked features reduced accuracy by 26%. Removing low-importance features sometimes improved results, confirming that noisy inputs can hurt model robustness.

6. **Emotion co-occurrence increases error rates**  
   Emotions that often appear alongside others (e.g., *surprise*, *fear*) are more likely to be missed. Top-k thresholding contributes to recall issues for minority emotions.

7. **Cultural and genre biases affect generalizability**  
   The dataset is biased toward Western, well-tagged music. Models often misinterpret emotions expressed through performance style or instrumentation in underrepresented genres.

---

Ciatation:
To cite this work, please use the following BibTeX entry:

```bibtex
@unpublished{zhong2025encoding,
  title={Encoding Emotion in Music via Acoustic Features: A Weakly Supervised Machine Learning Study},
  author={Jiaming Mao},
  note={Computational Social Science, University of Chicago},
  year={2025}
}
```

---

Reproducing Results:
To reproduce the results, follow these steps:
1. Clone the repository:
   ```bash
   git clone
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Spotify API credentials in `spotify_secret.py`.
4. Run the data collection scripts in `DataCollection&Pre` to gather Spotify audio features and Last.fm tags.
5. Preprocess the data using the scripts in `DataCollection&Pre`.
6. Train the models using the scripts in `ML_Progress`.
7. Evaluate the models and visualize the results using the scripts in `ML_Progress`.
8. Explore the results in the `Results` folder.
9. View the presentation slides in the `Presentation` folder.
10. Read the paper in the `Paper` folder.

---
## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact
For any questions or feedback, please contact Jiaming Mao at jmao0220@uchicago.edu


