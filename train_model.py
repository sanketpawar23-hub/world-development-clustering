"""
Train the clustering pipeline from the command line and save the results.

Usage:
    python train_model.py                              # K-Means, 4 clusters
    python train_model.py --algorithm "Gaussian Mixture" --clusters 3
"""
import argparse
from pathlib import Path

import joblib

import pipeline as pl

BASE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE / "data" / "World_development_mesurement.xlsx"))
    parser.add_argument("--algorithm", default="K-Means", choices=pl.ALGORITHMS)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--variance", type=float, default=0.90)
    args = parser.parse_args()

    raw = pl.load_raw(args.data)
    country_data = pl.prepare_country_data(raw)
    fit = pl.fit_pipeline(country_data, args.algorithm, args.clusters, args.variance)
    art, results = fit["artifacts"], fit["results"]

    out_dir = BASE / "outputs"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(art, out_dir / "clustering_artifacts.joblib")
    results.reset_index().to_csv(out_dir / "country_clusters.csv", index=False)

    print(f"Model: {art['algorithm']}  |  clusters: {art['n_clusters']}")
    print(f"Countries: {len(results)}  |  features: {len(art['feature_columns'])}  |  "
          f"PCA components: {art['pca'].n_components_} ({art['variance_retained']:.1%} variance)")
    for name, value in art["metrics"].items():
        print(f"{name}: {value:.3f}")
    print(results["Cluster Name"].value_counts().to_string())
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
