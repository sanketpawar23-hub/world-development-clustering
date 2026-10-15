# 🌍 World Development Clustering

Group countries by economic, health, demographic and technology indicators using unsupervised
machine learning, then explore the results in an interactive Streamlit app.

## What the app does
| Tab | What you get |
|---|---|
| 📊 Overview | Cluster sizes, PCA scatter plot, and a plain-English summary of what defines each cluster |
| 🗺️ World Map | Choropleth map coloured by cluster (or by any single indicator) |
| 🧬 Cluster Profiles | Heatmap of standardised cluster profiles, radar chart, median values per cluster |
| 🔎 Country Explorer | Pick a country: cluster, development score and rank, radar vs. its cluster, 5 most similar countries |
| ⚖️ Compare | Compare 2-5 countries on a radar chart, a bar chart and a full table |
| 🔮 Predict New Country | Enter a few indicators with sliders and get the predicted cluster, probabilities and similar countries |
| 🧪 Model Lab | Silhouette / Davies-Bouldin / Calinski-Harabasz for K-Means, Gaussian Mixture, Agglomerative and BIRCH |
| 📥 Data & Export | Download results (CSV) and the trained model (.joblib), data-quality chart, preprocessing steps |

The sidebar lets you choose the algorithm (K-Means, Gaussian Mixture, BIRCH), the number of clusters
(2-8), the PCA variance to keep, and whether to use the sample data or upload your own Excel/CSV.
Clusters are ranked automatically from least to most developed and given readable names.

## Project structure
```
app.py                 Streamlit user interface
pipeline.py            Cleaning, preprocessing, PCA, clustering, prediction helpers
country_codes.py       Country name -> ISO-3 code mapping for the world map
train_model.py         Command-line training script (saves outputs/)
data/                  World_development_mesurement.xlsx (sample data)
outputs/               clustering_artifacts.joblib, country_clusters.csv (created by train_model.py)
.streamlit/config.toml App theme
requirements.txt       Python dependencies
```

## Run it locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Optional: `python train_model.py --algorithm "K-Means" --clusters 4` saves the model and results to `outputs/`.

## Deploy for free (Streamlit Community Cloud)
1. Push this folder to a public GitHub repository.
2. Go to share.streamlit.io, sign in with GitHub and choose **New app**.
3. Select the repo, set the main file to `app.py`, and click **Deploy**.
4. Put the live link in this README and on your resume.

## Method
1. Convert text values such as `$1,234` and `12%` to numbers.
2. Take the median per country across years (2,704 rows -> 208 countries).
3. Drop indicators missing for more than 60% of countries and the useless `Number of Records` counter.
4. Median imputation, log(1+x) on skewed non-negative indicators (the list is saved with the model so new data is transformed identically), RobustScaler.
5. PCA keeping 90% of the variance (8 components on the sample data).
6. Clustering, then ranking clusters by a development score (life expectancy, infant mortality, internet and mobile use, health spending, urbanisation, birth rate).

## Improvements over the first version
- Selectable number of clusters (the original model produced only 2 clusters) and a Model Lab to justify the choice.
- Clusters are named by development level instead of 0, 1, 2.
- Fixed an inconsistency: the original app decided which columns to log-transform from the uploaded file; the list is now saved at training time.
- Removed the constant `Number of Records` feature.
- Interactive map, country comparison, similar-country search and a predict-new-country form.
- Sample data is bundled, so the app works immediately.

## Honest notes and limitations
- Silhouette scores are modest (about 0.22 for K-Means with 4 clusters, about 0.29 for 2 clusters). The data separate clearly into "developed" and "developing" groups; finer splits overlap more.
- Medians over all years hide trends over time; the dataset has no year column, so a trend analysis is not possible without adding one.
- Indicators entered on the Predict tab are partial; the rest are estimated from the 10 most similar countries.
- Kosovo and a few small territories may not appear on the map if the map library has no shape for them.
