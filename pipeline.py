"""
Core machine-learning pipeline for the World Development Clustering project.

Steps: clean -> country-level medians -> impute -> log transform -> robust scale
       -> PCA -> clustering -> rank clusters by development level -> name clusters.

Used by both app.py (Streamlit UI) and train_model.py (command line).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

# "Number of Records" is just a row counter (always 1 per country) - it carries no information.
DROP_COLUMNS = ["Number of Records"]
# Features that are missing for more than this share of countries are dropped.
MAX_MISSING = 0.60

# Algorithms that can assign NEW countries to clusters (needed for the "Predict" tab).
ALGORITHMS = ["K-Means", "Gaussian Mixture", "BIRCH"]

# Indicators used to rank clusters from "least" to "most" developed.
DEV_POSITIVE = [
    "Life Expectancy Female",
    "Life Expectancy Male",
    "Internet Usage",
    "Mobile Phone Usage",
    "Health Exp/Capita",
    "Population Urban",
]
DEV_NEGATIVE = ["Infant Mortality Rate", "Birth Rate"]

TIER_NAMES = {
    2: ["Developing", "Developed"],
    3: ["Low Development", "Middle Development", "High Development"],
    4: [
        "Low Development",
        "Lower-Middle Development",
        "Upper-Middle Development",
        "High Development",
    ],
}


# ----------------------------------------------------------------------------
# Loading and cleaning
# ----------------------------------------------------------------------------
def clean_numeric_value(value):
    """Turn strings like '$1,234' or '12%' into numbers (NaN if impossible)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip().replace("$", "").replace(",", "").replace("%", "")
    return pd.to_numeric(value, errors="coerce")


def load_raw(source, name=None):
    """Read an Excel or CSV file (path or file-like object)."""
    filename = str(name or source).lower()
    if filename.endswith(".csv"):
        return pd.read_csv(source)
    return pd.read_excel(source)


def prepare_country_data(raw, feature_columns=None, max_missing=MAX_MISSING):
    """
    Clean the raw multi-year data and aggregate it to one row per country (median).

    If feature_columns is given, exactly those columns are returned (used for new data);
    otherwise features are chosen automatically.
    """
    if "Country" not in raw.columns:
        raise ValueError("The dataset must contain a 'Country' column.")

    data = raw.copy()
    for column in data.columns:
        if column != "Country":
            data[column] = data[column].apply(clean_numeric_value)

    country_data = data.groupby("Country").median(numeric_only=True)
    country_data = country_data.drop(columns=DROP_COLUMNS, errors="ignore")
    country_data = country_data.replace([np.inf, -np.inf], np.nan)

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in country_data.columns]
        if missing:
            raise ValueError("Missing features: " + ", ".join(missing))
        return country_data[feature_columns]

    keep = [c for c in country_data.columns if country_data[c].isna().mean() <= max_missing]
    if len(keep) < 3:
        raise ValueError("Not enough usable numeric columns (need at least 3).")
    if len(country_data) < 10:
        raise ValueError("Need at least 10 countries to cluster.")
    return country_data[keep]


# ----------------------------------------------------------------------------
# Preprocessing + models
# ----------------------------------------------------------------------------
def preprocess_fit(country_data, variance=0.90):
    """Fit imputer, log transform, scaler and PCA on country-level data."""
    feature_columns = country_data.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    x_imp = pd.DataFrame(
        imputer.fit_transform(country_data),
        columns=feature_columns,
        index=country_data.index,
    )

    # Log-transform skewed, non-negative columns. The list is SAVED so that new data
    # is transformed exactly like the training data.
    log_columns = [c for c in feature_columns if (x_imp[c] >= 0).all()]
    x_trans = x_imp.copy()
    x_trans[log_columns] = np.log1p(x_trans[log_columns])

    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x_trans)

    full = PCA(random_state=RANDOM_STATE).fit(x_scaled)
    n_components = int(np.argmax(np.cumsum(full.explained_variance_ratio_) >= variance) + 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    x_pca = pca.fit_transform(x_scaled)

    return {
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "feature_columns": feature_columns,
        "log_columns": log_columns,
        "x_trans": x_trans,
        "x_pca": x_pca,
    }


def make_model(algorithm, n_clusters, random_state=RANDOM_STATE):
    if algorithm == "K-Means":
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    if algorithm == "Gaussian Mixture":
        return GaussianMixture(
            n_components=n_clusters,
            covariance_type="tied",
            random_state=random_state,
            n_init=5,
        )
    if algorithm == "BIRCH":
        return Birch(n_clusters=n_clusters, threshold=0.5, branching_factor=50)
    if algorithm == "Agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_labels(x, labels):
    """Silhouette / Davies-Bouldin / Calinski-Harabasz (NaN if fewer than 2 clusters)."""
    if len(np.unique(labels)) < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan}
    return {
        "Silhouette": float(silhouette_score(x, labels)),
        "Davies-Bouldin": float(davies_bouldin_score(x, labels)),
        "Calinski-Harabasz": float(calinski_harabasz_score(x, labels)),
    }


def development_score(x_trans):
    """Average z-score of development indicators (higher = more developed)."""
    std = x_trans.std(ddof=0).replace(0, 1)
    z = (x_trans - x_trans.mean()) / std
    parts = [z[c] for c in DEV_POSITIVE if c in z] + [-z[c] for c in DEV_NEGATIVE if c in z]
    if not parts:
        return None
    return pd.concat(parts, axis=1).mean(axis=1)


def tier_names(k):
    if k in TIER_NAMES:
        return TIER_NAMES[k]
    names = [f"Tier {i + 1}" for i in range(k)]
    names[0] += " (lowest)"
    names[-1] += " (highest)"
    return names


# ----------------------------------------------------------------------------
# Fit the full pipeline
# ----------------------------------------------------------------------------
def fit_pipeline(country_data, algorithm="K-Means", n_clusters=4, variance=0.90):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm must be one of {ALGORITHMS}")
    if not 2 <= n_clusters < len(country_data):
        raise ValueError("Number of clusters must be between 2 and (countries - 1).")

    prep = preprocess_fit(country_data, variance)
    x_pca = prep["x_pca"]

    model = make_model(algorithm, n_clusters)
    raw_labels = np.asarray(model.fit_predict(x_pca))
    present = np.unique(raw_labels)
    if len(present) < 2:
        raise ValueError("The model found only one cluster. Try a different setting.")

    # Rank clusters from least to most developed.
    score = development_score(prep["x_trans"])
    if score is not None:
        cluster_score = pd.Series(score.values).groupby(raw_labels).mean()
    else:  # fall back to the first principal component
        cluster_score = pd.Series(x_pca[:, 0]).groupby(raw_labels).mean()
    order = cluster_score.sort_values().index.tolist()  # raw labels, lowest -> highest
    label_map = {raw: rank for rank, raw in enumerate(order)}
    ranks = np.array([label_map[label] for label in raw_labels])

    names = tier_names(len(order))
    centroids = np.vstack([x_pca[raw_labels == raw].mean(axis=0) for raw in order])

    pca_columns = [f"PC{i + 1}" for i in range(x_pca.shape[1])]
    pca_df = pd.DataFrame(x_pca, columns=pca_columns, index=country_data.index)

    results = country_data.copy()
    results["Cluster"] = ranks
    results["Cluster Name"] = [names[r] for r in ranks]
    if score is not None:
        results["Development Score"] = score.round(3).values

    artifacts = {
        "imputer": prep["imputer"],
        "scaler": prep["scaler"],
        "pca": prep["pca"],
        "model": model,
        "algorithm": algorithm,
        "n_clusters": len(order),
        "feature_columns": prep["feature_columns"],
        "log_columns": prep["log_columns"],
        "order": order,
        "label_map": label_map,
        "cluster_names": names,
        "centroids": centroids,
        "metrics": evaluate_labels(x_pca, ranks),
        "variance_retained": float(prep["pca"].explained_variance_ratio_.sum()),
    }
    return {"artifacts": artifacts, "results": results, "pca_df": pca_df}


# ----------------------------------------------------------------------------
# Using a trained pipeline on new data
# ----------------------------------------------------------------------------
def transform_features(artifacts, df):
    """Apply impute -> log -> scale -> PCA exactly like in training."""
    cols = artifacts["feature_columns"]
    x = df.reindex(columns=cols).replace([np.inf, -np.inf], np.nan)
    x_imp = pd.DataFrame(artifacts["imputer"].transform(x), columns=cols, index=x.index)
    logs = artifacts["log_columns"]
    x_imp[logs] = np.log1p(x_imp[logs].clip(lower=0))
    return artifacts["pca"].transform(artifacts["scaler"].transform(x_imp))


def predict_clusters(artifacts, df):
    """Return (rank labels, membership scores in rank order, PCA coordinates)."""
    x_pca = transform_features(artifacts, df)
    model = artifacts["model"]
    raw = np.asarray(model.predict(x_pca))
    ranks = np.array([artifacts["label_map"][r] for r in raw])

    if hasattr(model, "predict_proba"):  # Gaussian Mixture: true probabilities
        scores = model.predict_proba(x_pca)[:, artifacts["order"]]
    else:  # K-Means / BIRCH: closeness to each centroid (soft-max of -distance)
        dist = np.linalg.norm(x_pca[:, None, :] - artifacts["centroids"][None, :, :], axis=2)
        weights = np.exp(-(dist - dist.min(axis=1, keepdims=True)))
        scores = weights / weights.sum(axis=1, keepdims=True)
    return ranks, scores, x_pca


def complete_profile(results, feature_columns, given, n_neighbors=10):
    """
    Build a full indicator row from a few user-supplied values.

    Indicators the user did not enter are estimated as the median of the countries that are
    most similar on the values that WERE entered (instead of using the world median, which
    would drag every prediction towards the middle).
    """
    given = {k: v for k, v in given.items() if k in feature_columns and pd.notna(v)}
    row = {c: np.nan for c in feature_columns}
    row.update(given)
    if not given:
        return pd.DataFrame([row])

    cols = list(given)
    ref = results[cols].fillna(results[cols].median())
    std = ref.std().replace(0, 1)
    distance = np.sqrt((((ref - pd.Series(given)) / std) ** 2).sum(axis=1))
    neighbours = distance.sort_values().head(n_neighbors).index
    estimate = results.loc[neighbours, feature_columns].median()
    for column in feature_columns:
        if column not in given:
            row[column] = estimate[column]
    return pd.DataFrame([row])


def nearest_countries(pca_df, point, n=5, exclude=None):
    """Countries closest to a point in PCA space."""
    dist = np.linalg.norm(pca_df.values - np.asarray(point).reshape(1, -1), axis=1)
    out = pd.Series(dist, index=pca_df.index).sort_values()
    if exclude is not None:
        out = out.drop(exclude, errors="ignore")
    return out.head(n)


# ----------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------
def compare_models(x_pca, ks=range(2, 9)):
    """Score several algorithms for several cluster counts (used in the Model Lab tab)."""
    rows = []
    for algorithm in ["K-Means", "Gaussian Mixture", "Agglomerative", "BIRCH"]:
        for k in ks:
            if k >= len(x_pca):
                continue
            labels = np.asarray(make_model(algorithm, k).fit_predict(x_pca))
            row = {"Algorithm": algorithm, "Clusters": k, **evaluate_labels(x_pca, labels)}
            rows.append(row)
    table = pd.DataFrame(rows).dropna()
    if table.empty:
        return table
    table["Combined Rank"] = (
        table["Silhouette"].rank(ascending=False, method="min")
        + table["Davies-Bouldin"].rank(ascending=True, method="min")
        + table["Calinski-Harabasz"].rank(ascending=False, method="min")
    )
    return table.sort_values(["Combined Rank", "Silhouette"], ascending=[True, False]).reset_index(drop=True)


def cluster_highlights(results, feature_columns, top=3):
    """For each cluster rank, the indicators that are most above / below the world median."""
    overall_median = results[feature_columns].median()
    overall_std = results[feature_columns].std().replace(0, 1)
    out = {}
    for rank, group in results.groupby("Cluster"):
        z = ((group[feature_columns].median() - overall_median) / overall_std).dropna()
        out[int(rank)] = {
            "high": z.sort_values(ascending=False).head(top).index.tolist(),
            "low": z.sort_values().head(top).index.tolist(),
        }
    return out


def percentile_table(results, feature_columns):
    """Each country's percentile (0-100) for every indicator - handy for radar charts."""
    return results[feature_columns].rank(pct=True) * 100
