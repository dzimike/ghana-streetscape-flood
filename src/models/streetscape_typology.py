"""K-means typology clustering on segment-level EfficientNet predictions.

Aggregates 15-label probability vectors by pano_id, runs k-means for k=2-8,
selects optimal k by silhouette coefficient, and saves cluster profiles.

Usage:
  python src/models/streetscape_typology.py           # auto-select optimal k
  python src/models/streetscape_typology.py --k 6     # force k=6 (Paper 3)

When --k is supplied the validation plot and heatmap are saved with a _k{k}
suffix (e.g. typology_cluster_validation_k6.png, typology_cluster_heatmap_k6.png)
and the cluster CSV is saved as streetscape_typology_clusters_k6.csv.

Outputs (auto mode):
  outputs/tables/streetscape_typology_clusters.csv
  outputs/tables/typology_cluster_profiles.csv
  outputs/figures/typology_cluster_validation.png
  outputs/figures/typology_cluster_heatmap.png

Outputs (--k 6 mode):
  outputs/tables/streetscape_typology_clusters_k6.csv
  outputs/tables/typology_cluster_profiles_k6.csv
  outputs/figures/typology_cluster_validation_k6.png
  outputs/figures/typology_cluster_heatmap_k6.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[2]
PREDS   = ROOT / "data/processed/efficientnet_predictions.parquet"
OUT_DIR = ROOT / "outputs/tables"
FIG_DIR = ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = [
    "enet_visible_drain_present",
    "enet_open_gutter_present",
    "enet_blocked_drain_present",
    "enet_stagnant_water_visible",
    "enet_poor_road_condition",
    "enet_heavy_impervious_surface",
    "enet_unpaved_shoulder",
    "enet_informal_structure_near_drainage",
    "enet_solid_waste_accumulation",
    "enet_visible_waterway_or_stream",
    "enet_low_lying_street_form",
    "enet_roadside_erosion",
    "enet_pedestrian_exposure",
    "enet_culvert_or_bridge_visible",
    "enet_no_visible_drainage",
]

SHORT = {c: c.replace("enet_", "").replace("_", " ") for c in LABEL_COLS}


def main():
    print("Loading predictions ...")
    df = pd.read_parquet(PREDS)
    print(f"  {len(df):,} image-level predictions, columns: {list(df.columns)}")

    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    # Aggregate to panorama (road-point) level by averaging across headings
    seg = (df.groupby("pano_id")[LABEL_COLS + ["enet_vuln_prob"]]
             .mean()
             .reset_index())
    print(f"  {len(seg):,} unique road-segment locations after aggregation")

    X  = seg[LABEL_COLS].values.astype(float)
    sc = StandardScaler()
    Xz = sc.fit_transform(X)

    # PCA summary
    pca = PCA(n_components=min(10, X.shape[1]))
    pca.fit(Xz)
    var_exp = np.cumsum(pca.explained_variance_ratio_)
    n_95 = int(np.searchsorted(var_exp, 0.95)) + 1
    print(f"\nPCA: {n_95} components explain 95% variance")
    for i, v in enumerate(var_exp[:6], 1):
        print(f"  PC{i}: {v:.3f} cumulative")

    # Cluster number selection
    ks, sil_scores, ch_scores = [], [], []
    print(f"\n  {'k':>3}  {'Silhouette':>12}  {'Calinski-Harabasz':>19}")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = km.fit_predict(Xz)
        sil = silhouette_score(Xz, labels,
                               sample_size=min(5000, len(Xz)), random_state=42)
        ch  = calinski_harabasz_score(Xz, labels)
        ks.append(k); sil_scores.append(sil); ch_scores.append(ch)
        print(f"  {k:>3}  {sil:>12.4f}  {ch:>19.1f}")

    best_k = ks[int(np.argmax(sil_scores))]
    print(f"\nOptimal k (max silhouette): {best_k}")

    # Final clustering
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=50, max_iter=1000)
    seg["cluster"] = km_final.fit_predict(Xz) + 1   # 1-indexed

    # Per-cluster silhouette for reporting
    from sklearn.metrics import silhouette_samples
    sil_vals = silhouette_samples(Xz, km_final.labels_)
    seg["silhouette"] = sil_vals

    # Cluster profiles
    profile_cols = LABEL_COLS + ["enet_vuln_prob", "silhouette"]
    profile = seg.groupby("cluster")[profile_cols].mean()
    profile["n"]   = seg.groupby("cluster").size()
    profile["pct"] = (profile["n"] / len(seg) * 100).round(1)
    profile = profile.sort_values("enet_vuln_prob", ascending=False)

    print("\nCluster profiles (mean probabilities, sorted by vulnerability):")
    print(profile[["n", "pct", "enet_vuln_prob", "silhouette"] + LABEL_COLS].round(3).to_string())

    # Save
    seg.to_csv(OUT_DIR / "streetscape_typology_clusters.csv", index=False)
    profile.round(3).to_csv(OUT_DIR / "typology_cluster_profiles.csv")
    print(f"\nSaved: {OUT_DIR}/streetscape_typology_clusters.csv")
    print(f"Saved: {OUT_DIR}/typology_cluster_profiles.csv")

    # Validation figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("white")

    axes[0].plot(ks, sil_scores, "o-", color="#2166ac", lw=2, ms=7)
    axes[0].axvline(best_k, color="#d7191c", ls="--", alpha=0.8, label=f"k={best_k}")
    axes[0].set_xlabel("Number of clusters k", fontsize=10)
    axes[0].set_ylabel("Mean silhouette coefficient", fontsize=10)
    axes[0].set_title("(a) Silhouette coefficient", fontsize=11)
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ks, ch_scores, "s-", color="#1a9641", lw=2, ms=7)
    axes[1].axvline(best_k, color="#d7191c", ls="--", alpha=0.8, label=f"k={best_k}")
    axes[1].set_xlabel("Number of clusters k", fontsize=10)
    axes[1].set_ylabel("Calinski-Harabasz index", fontsize=10)
    axes[1].set_title("(b) Calinski-Harabasz index", fontsize=11)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fpath = FIG_DIR / "typology_cluster_validation.png"
    plt.savefig(fpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fpath}")

    # Heatmap of cluster profiles
    fig2, ax = plt.subplots(figsize=(13, 4))
    fig2.patch.set_facecolor("white")
    mat = profile[LABEL_COLS].values
    im  = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABEL_COLS)))
    ax.set_xticklabels([SHORT[c] for c in LABEL_COLS],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"Cluster {int(c)}" for c in profile.index], fontsize=9)
    ax.set_title("Streetscape typology cluster profiles (mean label probability)",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label="Mean probability")
    plt.tight_layout()
    fpath2 = FIG_DIR / "typology_cluster_heatmap.png"
    plt.savefig(fpath2, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {fpath2}")

    return best_k, profile, ks, sil_scores, ch_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streetscape k-means typology")
    parser.add_argument("--k", type=int, default=None,
                        help="Force a specific k instead of auto-selecting by silhouette")
    args = parser.parse_args()

    best_k, profile, ks, sil_scores, ch_scores = main()

    if args.k is not None and args.k != best_k:
        k_forced = args.k
        print(f"\n── Forced k={k_forced} run ──────────────────────────────────────")
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_samples
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_parquet(PREDS)
        seg = (df.groupby("pano_id")[LABEL_COLS + ["enet_vuln_prob"]]
                 .mean().reset_index())
        X  = seg[LABEL_COLS].values.astype(float)
        sc = StandardScaler(); Xz = sc.fit_transform(X)
        km = KMeans(n_clusters=k_forced, random_state=42, n_init=50, max_iter=1000)
        seg["cluster"] = km.fit_predict(Xz) + 1
        seg["silhouette"] = silhouette_samples(Xz, km.labels_)

        profile_cols = LABEL_COLS + ["enet_vuln_prob", "silhouette"]
        prof = seg.groupby("cluster")[profile_cols].mean()
        prof["n"]   = seg.groupby("cluster").size()
        prof["pct"] = (prof["n"] / len(seg) * 100).round(1)
        prof = prof.sort_values("enet_vuln_prob", ascending=False)
        print(prof[["n", "pct", "enet_vuln_prob"] + LABEL_COLS].round(3).to_string())

        suffix = f"_k{k_forced}"
        seg.to_csv(OUT_DIR / f"streetscape_typology_clusters{suffix}.csv", index=False)
        prof.round(3).to_csv(OUT_DIR / f"typology_cluster_profiles{suffix}.csv")
        print(f"Saved: {OUT_DIR}/streetscape_typology_clusters{suffix}.csv")

        # Validation plot with both curves + forced k marker
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("white")
        axes[0].plot(ks, sil_scores, "o-", color="#2166ac", lw=2, ms=7)
        axes[0].axvline(best_k,    color="#d7191c", ls="--", alpha=0.6, label=f"optimal k={best_k}")
        axes[0].axvline(k_forced,  color="#756bb1", ls=":",  alpha=0.9, label=f"chosen k={k_forced}")
        axes[0].set_xlabel("k"); axes[0].set_ylabel("Mean silhouette")
        axes[0].set_title("(a) Silhouette coefficient"); axes[0].legend(); axes[0].grid(alpha=0.3)
        from sklearn.metrics import calinski_harabasz_score
        ch_scores2 = []
        for k in ks:
            _km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
            _labels = _km.fit_predict(Xz)
            ch_scores2.append(calinski_harabasz_score(Xz, _labels))
        axes[1].plot(ks, ch_scores2, "s-", color="#1a9641", lw=2, ms=7)
        axes[1].axvline(best_k,   color="#d7191c", ls="--", alpha=0.6, label=f"optimal k={best_k}")
        axes[1].axvline(k_forced, color="#756bb1", ls=":",  alpha=0.9, label=f"chosen k={k_forced}")
        axes[1].set_xlabel("k"); axes[1].set_ylabel("Calinski-Harabasz index")
        axes[1].set_title("(b) Calinski-Harabasz index"); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        vpath = FIG_DIR / f"typology_cluster_validation{suffix}.png"
        plt.savefig(vpath, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
        print(f"Saved: {vpath}")

        # Heatmap
        fig2, ax = plt.subplots(figsize=(13, max(4, k_forced * 0.7)))
        fig2.patch.set_facecolor("white")
        mat = prof[LABEL_COLS].values
        im  = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_xticks(range(len(LABEL_COLS)))
        ax.set_xticklabels([SHORT[c] for c in LABEL_COLS], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(k_forced))
        ax.set_yticklabels([f"Cluster {int(c)}" for c in prof.index], fontsize=9)
        ax.set_title(f"Streetscape typology — k={k_forced} cluster profiles (mean label probability)", fontsize=11)
        plt.colorbar(im, ax=ax, label="Mean probability"); plt.tight_layout()
        hpath = FIG_DIR / f"typology_cluster_heatmap{suffix}.png"
        plt.savefig(hpath, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
        print(f"Saved: {hpath}")
