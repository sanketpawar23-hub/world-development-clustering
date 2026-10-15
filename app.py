"""
World Development Clustering - interactive Streamlit app.

Run with:  streamlit run app.py
"""
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as pl
from country_codes import to_iso3

BASE = Path(__file__).parent
SAMPLE_PATH = BASE / "data" / "World_development_mesurement.xlsx"

# Indicators shown on radar charts / quick metrics (only those present in the data are used).
KEY_INDICATORS = [
    "Life Expectancy Female",
    "Infant Mortality Rate",
    "Internet Usage",
    "Mobile Phone Usage",
    "Health Exp/Capita",
    "Population Urban",
    "Birth Rate",
    "GDP",
    "CO2 Emissions",
    "Energy Usage",
]
# Indicators with friendly ranges that users can type in on the "Predict" tab.
PREDICT_INDICATORS = [
    "Life Expectancy Female",
    "Life Expectancy Male",
    "Infant Mortality Rate",
    "Birth Rate",
    "Internet Usage",
    "Mobile Phone Usage",
    "Health Exp/Capita",
    "Population Urban",
    "Population 0-14",
    "Population 65+",
]

st.set_page_config(page_title="World Development Clustering", page_icon="🌍", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #0f4c75 0%, #1b6ca8 50%, #3282b8 100%);
        color: #ffffff;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: #ffffff; }
    .hero p { margin: .4rem 0 0 0; font-size: 1.05rem; opacity: .92; color: #ffffff; }
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.25);
        padding: .8rem 1rem; border-radius: 12px;
    }
    .badge {
        display: inline-block; padding: .25rem .8rem; border-radius: 999px;
        color: #ffffff; font-weight: 600; font-size: .95rem;
    }
    .card {
        border: 1px solid rgba(128,128,128,0.30); border-radius: 12px;
        padding: .9rem 1rem; margin-bottom: .6rem; border-left-width: 6px;
    }
    .card h4 { margin: 0 0 .3rem 0; }
    .small { font-size: .88rem; opacity: .85; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _use_new_width_api():
    """Newer Streamlit (1.50+) uses width="stretch"; older versions use use_container_width."""
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        return (major, minor) >= (1, 50)
    except Exception:  # noqa: BLE001
        return False


NEW_WIDTH_API = _use_new_width_api()


def show_plot(fig, key):
    if NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(data, **kwargs):
    if NEW_WIDTH_API:
        st.dataframe(data, width="stretch", **kwargs)
    else:
        st.dataframe(data, use_container_width=True, **kwargs)


def format_frame(frame):
    """Apply fmt() to every cell (DataFrame.map needs pandas 2.1+, applymap is the old name)."""
    return frame.map(fmt) if hasattr(frame, "map") else frame.applymap(fmt)


def make_colors(k):
    """k colours from red (least developed) to green (most developed)."""
    stops = np.array(
        [[215, 48, 39], [244, 109, 67], [253, 174, 97], [166, 217, 106], [26, 152, 80]], dtype=float
    )
    positions = np.linspace(0, 1, len(stops))
    colors = []
    for t in np.linspace(0, 1, k):
        rgb = [int(np.interp(t, positions, stops[:, i])) for i in range(3)]
        colors.append("#%02x%02x%02x" % tuple(rgb))
    return colors


def fmt(value):
    """Readable numbers (1.2B, 35.4M, 0.31...)."""
    if value is None or pd.isna(value):
        return "n/a"
    a = abs(value)
    for limit, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if a >= limit:
            return f"{value / limit:.1f}{suffix}"
    return f"{value:.3f}" if a < 1 else f"{value:.1f}"


def radar(percentiles, labels_to_series, colors, title):
    """Radar chart; percentiles is a DataFrame (rows=entities) of 0-100 values."""
    fig = go.Figure()
    axes = percentiles.columns.tolist()
    for label in labels_to_series:
        values = percentiles.loc[label, axes].fillna(0).tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=axes + axes[:1],
                name=str(label),
                fill="toself",
                opacity=0.55,
                line=dict(color=colors.get(label)) if colors.get(label) else None,
            )
        )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True)),
        margin=dict(l=40, r=40, t=60, b=30),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


# ----------------------------------------------------------------------------
# Data loading + model (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sample():
    return pl.load_raw(SAMPLE_PATH)


@st.cache_data(show_spinner=False)
def read_upload(name, content):
    return pl.load_raw(io.BytesIO(content), name)


@st.cache_resource(show_spinner="Training clustering model...")
def run_pipeline(raw, algorithm, n_clusters, variance):
    country_data = pl.prepare_country_data(raw)
    return pl.fit_pipeline(country_data, algorithm, n_clusters, variance)


@st.cache_data(show_spinner="Comparing algorithms...")
def run_comparison(x_pca):
    return pl.compare_models(x_pca)


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
source = st.sidebar.radio("Data source", ["Sample dataset", "Upload my own file"])

raw_data = None
if source == "Upload my own file":
    uploaded = st.sidebar.file_uploader("Excel or CSV with a 'Country' column", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        try:
            raw_data = read_upload(uploaded.name, uploaded.getvalue())
        except Exception as error:  # noqa: BLE001
            st.sidebar.error(f"Could not read the file: {error}")
    if raw_data is None:
        st.sidebar.info("No valid file yet - showing the sample dataset.")
if raw_data is None:
    raw_data = load_sample()

algorithm = st.sidebar.selectbox("Algorithm", pl.ALGORITHMS, index=0)
n_clusters = st.sidebar.slider("Number of clusters", 2, 8, 4)
with st.sidebar.expander("Advanced"):
    variance = st.slider("PCA variance to keep", 0.70, 0.99, 0.90, 0.01)

st.sidebar.caption(
    "Clusters are automatically ranked and named from least to most developed "
    "(using life expectancy, infant mortality, internet use, and similar indicators)."
)

try:
    fit = run_pipeline(raw_data, algorithm, n_clusters, variance)
except Exception as error:  # noqa: BLE001
    st.error(f"Could not build the clusters: {error}")
    st.stop()

art = fit["artifacts"]
res = fit["results"]
pca_df = fit["pca_df"]
features = art["feature_columns"]
names = art["cluster_names"]
palette = make_colors(len(names))
color_map = dict(zip(names, palette))
key_cols = [c for c in KEY_INDICATORS if c in features]
pct = pl.percentile_table(res, features)

# ----------------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🌍 World Development Clustering</h1>
      <p>Group countries by economic, health, demographic and technology indicators
      using unsupervised machine learning - then explore, compare and predict.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Countries", len(res))
k2.metric("Clusters", art["n_clusters"])
k3.metric("Indicators", len(features))
k4.metric("PCA components", f"{art['pca'].n_components_} ({art['variance_retained']:.0%})")
k5.metric("Silhouette", f"{art['metrics']['Silhouette']:.2f}")

tabs = st.tabs(
    [
        "📊 Overview",
        "🗺️ World Map",
        "🧬 Cluster Profiles",
        "🔎 Country Explorer",
        "⚖️ Compare",
        "🔮 Predict New Country",
        "🧪 Model Lab",
        "📥 Data & Export",
    ]
)

# ----------------------------------------------------------------------------
# 1. Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([1, 1.4])
    counts = res["Cluster Name"].value_counts().reindex(names).fillna(0).reset_index()
    counts.columns = ["Cluster", "Countries"]

    with left:
        fig = px.pie(
            counts, names="Cluster", values="Countries", hole=0.55,
            color="Cluster", color_discrete_map=color_map, category_orders={"Cluster": names},
        )
        fig.update_layout(title="Countries per cluster", margin=dict(t=60, b=10))
        show_plot(fig, "overview_pie")

    with right:
        plot_df = pca_df[["PC1", "PC2"]].copy()
        plot_df["Cluster"] = res["Cluster Name"]
        plot_df["Country"] = plot_df.index
        fig = px.scatter(
            plot_df, x="PC1", y="PC2", color="Cluster", hover_name="Country",
            color_discrete_map=color_map, category_orders={"Cluster": names},
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
        fig.update_layout(title="Countries in PCA space (PC1 vs PC2)", margin=dict(t=60, b=10))
        show_plot(fig, "overview_scatter")

    st.subheader("What defines each cluster?")
    highlights = pl.cluster_highlights(res, features)
    for rank, name in enumerate(names):
        info = highlights.get(rank)
        if info is None:
            continue
        members = res.index[res["Cluster"] == rank].tolist()
        st.markdown(
            f"""
            <div class="card" style="border-left-color:{palette[rank]}">
              <h4>{name} &nbsp;<span class="small">({len(members)} countries)</span></h4>
              <div class="small">⬆️ Highest vs world: <b>{", ".join(info["high"])}</b></div>
              <div class="small">⬇️ Lowest vs world: <b>{", ".join(info["low"])}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Show countries in '{name}'"):
            st.write(", ".join(sorted(members)))

# ----------------------------------------------------------------------------
# 2. World map
# ----------------------------------------------------------------------------
with tabs[1]:
    map_df = res.copy()
    map_df["Country"] = map_df.index
    map_df["ISO3"] = map_df["Country"].map(to_iso3)
    drawable = map_df.dropna(subset=["ISO3"])
    if len(drawable) < len(map_df):
        st.caption(f"{len(map_df) - len(drawable)} countries could not be matched to a map code and are not drawn.")

    colour_by = st.selectbox("Colour the map by", ["Cluster"] + features, index=0)
    if colour_by == "Cluster":
        fig = px.choropleth(
            drawable, locations="ISO3", color="Cluster Name", hover_name="Country",
            color_discrete_map=color_map, category_orders={"Cluster Name": names},
            hover_data={"ISO3": False, "Cluster Name": True},
        )
    else:
        fig = px.choropleth(
            drawable, locations="ISO3", color=colour_by, hover_name="Country",
            color_continuous_scale="Viridis", hover_data={"ISO3": False},
        )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=560, legend_title_text="")
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth")
    show_plot(fig, "world_map")

# ----------------------------------------------------------------------------
# 3. Cluster profiles
# ----------------------------------------------------------------------------
with tabs[2]:
    medians = res.groupby("Cluster Name")[features].median().reindex(names)
    overall_std = res[features].std().replace(0, 1)
    z = (medians - res[features].median()) / overall_std

    fig = px.imshow(
        z.T, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
        labels=dict(color="Std. devs from world median"),
    )
    fig.update_layout(title="Standardised cluster profiles (red = above world median, blue = below)",
                      height=620, margin=dict(t=60))
    show_plot(fig, "profile_heatmap")

    if key_cols:
        cluster_pct = pd.DataFrame(
            {n: pct.loc[res["Cluster Name"] == n, key_cols].mean() for n in names}
        ).T
        fig = radar(cluster_pct, names, color_map, "Average percentile on key indicators")
        show_plot(fig, "profile_radar")

    st.subheader("Median values per cluster")
    show_df(format_frame(medians.T))

# ----------------------------------------------------------------------------
# 4. Country explorer
# ----------------------------------------------------------------------------
with tabs[3]:
    country_list = res.index.tolist()
    default_index = country_list.index("India") if "India" in country_list else 0
    country = st.selectbox("Select a country", country_list, index=default_index)
    row = res.loc[country]
    cname = row["Cluster Name"]

    st.markdown(
        f'<span class="badge" style="background:{color_map[cname]}">{cname}</span>',
        unsafe_allow_html=True,
    )
    if "Development Score" in res.columns:
        rank_pos = int(res["Development Score"].rank(ascending=False)[country])
        st.caption(f"Development score {row['Development Score']:.2f} - ranked #{rank_pos} of {len(res)}")

    shown = [c for c in key_cols if c in ("Life Expectancy Female", "Infant Mortality Rate",
                                          "Internet Usage", "Health Exp/Capita")] or key_cols[:4]
    for column, holder in zip(shown, st.columns(max(len(shown), 1))):
        holder.metric(column, fmt(row[column]))

    c1, c2 = st.columns([1.1, 1])
    with c1:
        if key_cols:
            radar_df = pct[key_cols].copy()
            radar_df.loc["Cluster average"] = pct.loc[res["Cluster Name"] == cname, key_cols].mean()
            fig = radar(radar_df, [country, "Cluster average"],
                        {country: "#1b6ca8", "Cluster average": "#999999"},
                        f"{country} vs its cluster (percentiles)")
            show_plot(fig, "explorer_radar")
    with c2:
        st.markdown("**Most similar countries**")
        near = pl.nearest_countries(pca_df, pca_df.loc[country].values, n=5, exclude=country)
        sim = pd.DataFrame({"Country": near.index, "Cluster": res.loc[near.index, "Cluster Name"].values,
                            "Distance": near.values.round(2)})
        show_df(sim, hide_index=True)

    with st.expander("All indicators for this country"):
        detail = row[features].to_frame("Value")
        detail["Value"] = detail["Value"].map(fmt)
        show_df(detail)

# ----------------------------------------------------------------------------
# 5. Compare
# ----------------------------------------------------------------------------
with tabs[4]:
    defaults = [c for c in ["India", "China", "United States"] if c in res.index] or res.index[:3].tolist()
    chosen = st.multiselect("Pick 2-5 countries", res.index.tolist(), default=defaults, max_selections=5)
    if len(chosen) < 2:
        st.info("Select at least two countries to compare.")
    else:
        if key_cols:
            fig = radar(pct[key_cols], chosen, {}, "Percentile on key indicators (100 = highest of all countries)")
            show_plot(fig, "compare_radar")
        indicator = st.selectbox("Compare one indicator", features, key="compare_indicator")
        bar_df = res.loc[chosen, [indicator, "Cluster Name"]].reset_index()
        bar_df.columns = ["Country", indicator, "Cluster Name"]
        fig = px.bar(bar_df, x="Country", y=indicator, color="Cluster Name",
                     color_discrete_map=color_map, category_orders={"Cluster Name": names})
        fig.update_layout(margin=dict(t=30))
        show_plot(fig, "compare_bar")
        show_df(format_frame(res.loc[chosen, features].T))

# ----------------------------------------------------------------------------
# 6. Predict a new country
# ----------------------------------------------------------------------------
with tabs[5]:
    st.write(
        "Enter indicators for a hypothetical (or real) country. Any indicator you do not set "
        "(GDP, CO2, tourism, ...) is estimated from the most similar countries, then the trained "
        "model assigns the country to a cluster."
    )
    form_cols = [c for c in PREDICT_INDICATORS if c in features]
    with st.form("predict_form"):
        inputs = {}
        grid = st.columns(2)
        for i, column in enumerate(form_cols):
            low, high = float(res[column].quantile(0.01)), float(res[column].quantile(0.99))
            if high <= low:
                high = low + 1.0
            median = float(np.clip(res[column].median(), low, high))
            step = (high - low) / 200
            text_format = "%.3f" if (high - low) < 10 else "%.1f"
            with grid[i % 2]:
                inputs[column] = st.slider(column, low, high, median, step, format=text_format)
        submitted = st.form_submit_button("Predict cluster")

    if submitted:
        new_row = pl.complete_profile(res, features, inputs)
        ranks, scores, point = pl.predict_clusters(art, new_row)
        predicted = names[int(ranks[0])]
        st.markdown(
            f'### Predicted cluster: <span class="badge" style="background:{color_map[predicted]}">{predicted}</span>',
            unsafe_allow_html=True,
        )
        label = "Membership probability" if hasattr(art["model"], "predict_proba") else "Closeness to each cluster"
        prob_df = pd.DataFrame({"Cluster": names, label: scores[0]})
        fig = px.bar(prob_df, x="Cluster", y=label, color="Cluster", color_discrete_map=color_map,
                     category_orders={"Cluster": names})
        fig.update_layout(showlegend=False, yaxis_range=[0, 1], margin=dict(t=30))
        show_plot(fig, "predict_bar")

        near = pl.nearest_countries(pca_df, point[0], n=5)
        st.markdown("**Most similar real countries**")
        show_df(
            pd.DataFrame({"Country": near.index, "Cluster": res.loc[near.index, "Cluster Name"].values}),
            hide_index=True,
        )

# ----------------------------------------------------------------------------
# 7. Model lab
# ----------------------------------------------------------------------------
with tabs[6]:
    st.write(
        "How do different algorithms and cluster counts score? Higher **Silhouette** and "
        "**Calinski-Harabasz** are better; lower **Davies-Bouldin** is better. "
        "Remember that the best score is not always the most useful number of clusters."
    )
    comparison = run_comparison(fit["pca_df"].values)
    if comparison.empty:
        st.info("Not enough data to compare models.")
    else:
        fig = px.line(comparison.sort_values("Clusters"), x="Clusters", y="Silhouette",
                      color="Algorithm", markers=True)
        fig.update_layout(title="Silhouette score by number of clusters", margin=dict(t=60))
        show_plot(fig, "lab_silhouette")
        st.markdown("**Ranking (combined rank of the three metrics - lower is better)**")
        show_df(comparison.round(3), hide_index=True)
    st.caption(
        f"Current model: {art['algorithm']} with {art['n_clusters']} clusters - "
        f"Silhouette {art['metrics']['Silhouette']:.3f}, Davies-Bouldin {art['metrics']['Davies-Bouldin']:.3f}, "
        f"Calinski-Harabasz {art['metrics']['Calinski-Harabasz']:.1f}."
    )

# ----------------------------------------------------------------------------
# 8. Data & export
# ----------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Cluster assignments")
    export = res.reset_index().rename(columns={"index": "Country"})
    show_df(export[["Country", "Cluster", "Cluster Name"] + [c for c in ["Development Score"] if c in export]]
            .sort_values("Cluster"), hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download results (CSV)", export.to_csv(index=False).encode("utf-8"),
                       "country_cluster_results.csv", "text/csv")
    buffer = io.BytesIO()
    joblib.dump(art, buffer)
    d2.download_button("⬇️ Download trained model (.joblib)", buffer.getvalue(),
                       "clustering_artifacts.joblib", "application/octet-stream")

    st.subheader("Data quality")
    missing = (res[features].isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.write("No missing values at country level.")
    else:
        missing_df = missing.rename("Missing %").rename_axis("Indicator").reset_index()
        fig = px.bar(missing_df, x="Indicator", y="Missing %")
        fig.update_layout(yaxis_title="% of countries missing", xaxis_title="", margin=dict(t=20))
        show_plot(fig, "missing_bar")
        st.caption("Missing values are filled with the median before clustering.")

    with st.expander("Preprocessing steps"):
        st.markdown(
            f"""
            1. Convert text values like `$1,234` or `12%` to numbers.
            2. Take the **median per country** across all years in the file.
            3. Drop indicators missing for more than {int(pl.MAX_MISSING * 100)}% of countries
               (and the useless `Number of Records` counter).
            4. Fill remaining gaps with the median, apply **log(1+x)** to {len(art["log_columns"])} skewed
               non-negative indicators, then scale with a **RobustScaler**.
            5. Reduce to **{art["pca"].n_components_} principal components** ({art["variance_retained"]:.1%} of variance).
            6. Cluster with **{art["algorithm"]}**, then rank clusters by a development score
               and name them.
            """
        )
    with st.expander("Raw data preview"):
        show_df(raw_data.head(50))

st.caption("Built with Python, scikit-learn, Plotly and Streamlit.")
