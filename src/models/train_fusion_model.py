"""Fusion model: combine streetscape + geospatial features → SLFVI — Task 9.

Workflow
--------
1. Merge streetscape_features + geospatial_features on point_id
2. Build SLFVI components (H, E, S, A) and compute raw index
3. Train ML classifiers (flood-exposed binary prediction):
     - Random Forest (always available)
     - XGBoost / LightGBM (requires libomp — skipped if missing)
4. Spatial block cross-validation (5-fold grid blocks)
5. SHAP feature importance
6. Export vulnerability_index.parquet + .gpkg + model comparison table

Target variable: any_flood_exposed_100m  (1 = ≥1 flood-exposed building within 100 m)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
# Use v2 if EfficientNet inference has been run, otherwise fall back to v1
_SF_V2 = ROOT / "data/processed/streetscape_features_v2.parquet"
_SF_V1 = ROOT / "data/processed/streetscape_features.parquet"
STREETSCAPE = _SF_V2 if _SF_V2.exists() else _SF_V1
GEOSPATIAL  = ROOT / "data/processed/geospatial_features.parquet"
SAMPLE_PTS  = ROOT / "data/interim/sample_points.gpkg"

OUT_PARQUET   = ROOT / "data/processed/vulnerability_index.parquet"
OUT_GPKG      = ROOT / "data/processed/vulnerability_index.gpkg"
OUT_MODEL_CMP = ROOT / "outputs/tables/model_comparison.csv"
OUT_SHAP      = ROOT / "outputs/tables/shap_feature_importance.csv"
OUT_PRIORITY  = ROOT / "outputs/tables/drainage_maintenance_priority.csv"


# ── Normalise to 0–1 ──────────────────────────────────────────────────────────
def _norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)


# ── SLFVI component builder ───────────────────────────────────────────────────
def build_slfvi(df: pd.DataFrame) -> pd.DataFrame:
    """Construct H, E, S, A components and final SLFVI score."""

    # ── H: Hazard ─────────────────────────────────────────────────────────
    # Low elevation, flat/depressed terrain, proximity to water = higher hazard
    h_elevation  = _norm(-df["elevation_m"].fillna(df["elevation_m"].median()))
    h_depression = _norm(-df["local_depression_m"].fillna(0))   # more negative = more depressed
    h_slope      = _norm(-df["slope_deg"].fillna(0))            # flatter = more pooling
    h_drain_prox = _norm(-df["dist_drain_m"].fillna(df["dist_drain_m"].median()))
    h_water_prox = _norm(-df["dist_waterway_m"].fillna(df["dist_waterway_m"].median()))

    df["H"] = (
        0.30 * h_elevation
        + 0.25 * h_depression
        + 0.15 * h_slope
        + 0.15 * h_drain_prox
        + 0.15 * h_water_prox
    )

    # ── E: Exposure ───────────────────────────────────────────────────────
    e_buildings = _norm(df["building_count_100m"].fillna(0))
    e_pop       = _norm(df["pop_density_per_km2"].fillna(0))
    e_flood_bld = _norm(df["flood_exposed_count_100m"].fillna(0))

    df["E"] = (
        0.35 * e_flood_bld
        + 0.35 * e_buildings
        + 0.30 * e_pop
    )

    # ── S: Streetscape Sensitivity ────────────────────────────────────────
    # Use EfficientNet-derived score (v2) when available, else CLIP-based score
    if "enet_sensitivity" in df.columns:
        s_col = "enet_sensitivity"
    elif "streetscape_sensitivity_score" in df.columns:
        s_col = "streetscape_sensitivity_score"
    else:
        s_col = None

    if s_col:
        df["S"] = df[s_col].fillna(df[s_col].median())
    else:
        df["S"] = 0.5

    # ── A: Adaptive Capacity (inverse — low capacity = high vulnerability) ─
    if "adaptive_capacity_raw_score" in df.columns:
        df["A"] = 1.0 - df["adaptive_capacity_raw_score"].fillna(
            df["adaptive_capacity_raw_score"].median()
        )
    else:
        # Proxy from EfficientNet road quality signal
        road_ok = df.get("enet_poor_road_condition", pd.Series(0.5, index=df.index))
        df["A"] = road_ok.fillna(0.5)

    # ── SLFVI = 0.30H + 0.20E + 0.35S + 0.15A ────────────────────────────
    df["slfvi"] = (
        0.30 * df["H"]
        + 0.20 * df["E"]
        + 0.35 * df["S"]
        + 0.15 * df["A"]
    )
    df["slfvi"] = _norm(df["slfvi"])

    # Classify
    bins   = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
    labels = ["very_low", "low", "moderate", "high", "very_high"]
    df["vulnerability_class"] = pd.cut(df["slfvi"], bins=bins, labels=labels)

    return df


# ── Feature matrix ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Terrain / hazard
    "elevation_m", "slope_deg", "local_depression_m",
    "dist_drain_m", "dist_waterway_m",
    # Exposure
    "building_count_100m", "building_count_250m", "pop_density_per_km2",
    # Streetscape sensitivity (v1 CLIP-based, kept for backward compat)
    "streetscape_sensitivity_score", "drain_detection_score",
    "drain_obstruction_score", "waste_risk_score", "water_exposure_score",
    "road_quality_score", "informal_encroachment_score", "low_lying_score",
    # Segmentation fractions
    "seg_road", "seg_water", "seg_vegetation", "seg_bare_ground",
    "seg_building", "seg_waste",
    # CLIP top signals
    "clip_blocked_drain_present", "clip_stagnant_water_visible",
    "clip_solid_waste_accumulation", "clip_poor_road_condition",
    "clip_visible_drain_present", "clip_no_visible_drainage",
    "clip_unpaved_shoulder", "clip_low_lying_street_form",
    # EfficientNet gold model (v2) — main flood indicator probabilities
    "enet_sensitivity", "enet_vuln_prob",
    "enet_visible_drain_present", "enet_open_gutter_present",
    "enet_blocked_drain_present", "enet_stagnant_water_visible",
    "enet_poor_road_condition", "enet_heavy_impervious_surface",
    "enet_solid_waste_accumulation", "enet_informal_structure_near_drainage",
    "enet_low_lying_street_form", "enet_no_visible_drainage",
]

TARGET_COL = "any_flood_exposed_100m"


# ── Spatial block cross-validation ───────────────────────────────────────────
def spatial_block_cv(df: pd.DataFrame, n_blocks: int = 5) -> list[tuple]:
    """Split points into spatial grid blocks for CV."""
    df = df.copy()
    lat_bins = pd.qcut(df["latitude"],  q=n_blocks, labels=False, duplicates="drop")
    lon_bins = pd.qcut(df["longitude"], q=n_blocks, labels=False, duplicates="drop")
    df["block"] = lat_bins.astype(str) + "_" + lon_bins.astype(str)

    blocks = df["block"].unique()
    np.random.seed(42)
    np.random.shuffle(blocks)

    # Group blocks into 5 folds
    fold_size = max(1, len(blocks) // n_blocks)
    folds = []
    for i in range(n_blocks):
        test_blocks  = blocks[i * fold_size: (i + 1) * fold_size]
        test_idx  = df[df["block"].isin(test_blocks)].index.tolist()
        train_idx = df[~df["block"].isin(test_blocks)].index.tolist()
        if test_idx and train_idx:
            folds.append((train_idx, test_idx))
    return folds


# ── Train + evaluate one model ────────────────────────────────────────────────
def train_evaluate(model, X: pd.DataFrame, y: pd.Series, folds: list) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    aucs, aps, f1s = [], [], []
    for train_idx, test_idx in folds:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if y_te.nunique() < 2:
            continue
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred  = model.predict(X_te)
        aucs.append(roc_auc_score(y_te, proba))
        aps.append(average_precision_score(y_te, proba))
        f1s.append(f1_score(y_te, pred, zero_division=0))

    return {
        "roc_auc":  round(np.mean(aucs), 4) if aucs else np.nan,
        "pr_auc":   round(np.mean(aps),  4) if aps  else np.nan,
        "f1":       round(np.mean(f1s),  4) if f1s  else np.nan,
        "n_folds":  len(aucs),
    }


def main() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    import geopandas as gpd

    # ── 1. Load and merge ─────────────────────────────────────────────────
    logger.info("Loading feature tables …")
    ss = pd.read_parquet(STREETSCAPE)
    gs = pd.read_parquet(GEOSPATIAL)
    logger.info(f"  Streetscape: {len(ss):,} points | Geospatial: {len(gs):,} points")

    # Drop columns from GS that are already in SS (v2 already joined geo features)
    gs_only_cols = [c for c in gs.columns
                    if c not in ss.columns or c in ["point_id", "highway",
                                                    "flood_stratum", "latitude", "longitude"]]
    gs_clean = gs[gs_only_cols]

    df = ss.merge(gs_clean, on=["point_id", "highway", "flood_stratum",
                                 "latitude", "longitude"], how="inner")

    # Resolve any remaining _x/_y suffix columns (prefer _x = streetscape source)
    for col in [c[:-2] for c in df.columns if c.endswith("_x")]:
        if f"{col}_x" in df.columns and f"{col}_y" in df.columns:
            df[col] = df[f"{col}_x"].combine_first(df[f"{col}_y"])
            df = df.drop(columns=[f"{col}_x", f"{col}_y"])

    logger.info(f"  Merged: {len(df):,} points with both feature sets")

    # ── 2. Build SLFVI ────────────────────────────────────────────────────
    logger.info("Computing SLFVI components …")
    df = build_slfvi(df)

    # ── 3. Prepare ML feature matrix ──────────────────────────────────────
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Missing feature columns (skipped): {missing}")

    X_raw = df[available].copy()
    y     = df[TARGET_COL].fillna(0).astype(int)

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=available, index=df.index)

    logger.info(f"  Features: {len(available)} | Target: {TARGET_COL} | "
                f"Positive rate: {y.mean():.1%}")

    # ── 4. Spatial CV folds ───────────────────────────────────────────────
    folds = spatial_block_cv(df)
    logger.info(f"  Spatial CV: {len(folds)} folds")

    # ── 5. Train models ───────────────────────────────────────────────────
    models = {}
    results = {}

    logger.info("Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    results["RandomForest"] = train_evaluate(rf, X, y, folds)
    rf.fit(X, y)   # final fit on all data
    models["RandomForest"] = rf
    logger.info(f"  RF  ROC-AUC={results['RandomForest']['roc_auc']} "
                f"PR-AUC={results['RandomForest']['pr_auc']}")

    # XGBoost — optional (requires libomp; install via: conda install -c conda-forge xgboost)
    try:
        import xgboost as xgb
        logger.info("Training XGBoost …")
        scale_pos = int((y == 0).sum() / max((y == 1).sum(), 1))
        xgb_m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos, eval_metric="logloss",
            random_state=42, verbosity=0
        )
        results["XGBoost"] = train_evaluate(xgb_m, X, y, folds)
        xgb_m.fit(X, y)
        models["XGBoost"] = xgb_m
        logger.info(f"  XGB ROC-AUC={results['XGBoost']['roc_auc']} "
                    f"PR-AUC={results['XGBoost']['pr_auc']}")
    except Exception as e:
        logger.warning(f"XGBoost skipped: {e}")

    # LightGBM — optional
    try:
        import lightgbm as lgb
        logger.info("Training LightGBM …")
        lgb_m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1
        )
        results["LightGBM"] = train_evaluate(lgb_m, X, y, folds)
        lgb_m.fit(X, y)
        models["LightGBM"] = lgb_m
        logger.info(f"  LGB ROC-AUC={results['LightGBM']['roc_auc']} "
                    f"PR-AUC={results['LightGBM']['pr_auc']}")
    except Exception as e:
        logger.warning(f"LightGBM skipped: {e}")

    # Best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"] or 0)
    best_model = models[best_name]
    logger.info(f"  Best model: {best_name}")

    # Save best model and feature list for downstream prediction
    import pickle
    MODEL_STORE = ROOT / "models/fusion"
    MODEL_STORE.mkdir(parents=True, exist_ok=True)
    with open(MODEL_STORE / "lgb.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_STORE / "feature_cols.pkl", "wb") as f:
        pickle.dump(available, f)
    logger.info(f"  Saved model → {MODEL_STORE}/lgb.pkl  ({len(available)} features)")

    # ── 6. ML vulnerability score ─────────────────────────────────────────
    df["ml_flood_prob"] = best_model.predict_proba(X)[:, 1]

    # Blend SLFVI with ML score (70% SLFVI formula, 30% ML)
    df["slfvi_final"] = _norm(
        0.70 * df["slfvi"] + 0.30 * _norm(df["ml_flood_prob"])
    )
    bins   = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
    labels = ["very_low", "low", "moderate", "high", "very_high"]
    df["vulnerability_class_final"] = pd.cut(
        df["slfvi_final"], bins=bins, labels=labels
    )

    # ── 7. SHAP feature importance ────────────────────────────────────────
    try:
        import shap
        logger.info("Computing SHAP values …")
        explainer  = shap.TreeExplainer(best_model)
        shap_vals  = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # positive class
        shap_mean  = np.abs(shap_vals).mean(axis=0)
        shap_df    = pd.DataFrame({
            "feature":         available,
            "shap_importance": shap_mean,
        }).sort_values("shap_importance", ascending=False)

        OUT_SHAP.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(OUT_SHAP, index=False)
        logger.info(f"SHAP → {OUT_SHAP}")
    except Exception as e:
        logger.warning(f"SHAP skipped: {e}")
        shap_df = None

    # ── 8. Export outputs ─────────────────────────────────────────────────
    # Model comparison table
    cmp_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    OUT_MODEL_CMP.parent.mkdir(parents=True, exist_ok=True)
    cmp_df.to_csv(OUT_MODEL_CMP, index=False)

    # Drainage maintenance priority (top 100 segments by SLFVI + obstruction)
    obstruction = df.get("drain_obstruction_score",
                         df.get("enet_blocked_drain_present", pd.Series(0, index=df.index)))
    waste_risk  = df.get("waste_risk_score",
                         df.get("enet_solid_waste_accumulation", pd.Series(0, index=df.index)))
    df["drain_obstruction_score"] = obstruction.fillna(0)
    df["waste_risk_score"]        = waste_risk.fillna(0)
    df["maintenance_priority_score"] = (
        0.50 * df["slfvi_final"]
        + 0.30 * df["drain_obstruction_score"]
        + 0.20 * df["waste_risk_score"]
    )
    priority = (
        df[["point_id", "highway", "flood_stratum", "latitude", "longitude",
            "slfvi_final", "drain_obstruction_score", "waste_risk_score",
            "maintenance_priority_score", "vulnerability_class_final"]]
        .sort_values("maintenance_priority_score", ascending=False)
        .head(100)
        .reset_index(drop=True)
    )
    priority.index += 1
    priority.index.name = "rank"
    priority.to_csv(OUT_PRIORITY)
    logger.info(f"Priority list → {OUT_PRIORITY}")

    # GeoPackage
    sp = gpd.read_file(SAMPLE_PTS)[["point_id", "geometry"]]
    export_cols = [
        "point_id", "highway", "flood_stratum", "latitude", "longitude",
        "H", "E", "S", "A", "slfvi", "slfvi_final",
        "vulnerability_class", "vulnerability_class_final",
        "ml_flood_prob", "maintenance_priority_score",
        "streetscape_sensitivity_score", "enet_sensitivity", "enet_vuln_prob",
        "drain_obstruction_score", "waste_risk_score",
        "elevation_m", "slope_deg", "local_depression_m",
        "building_count_100m", "flood_exposed_count_100m", "pop_density_per_km2",
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    gdf = sp.merge(df[export_cols], on="point_id", how="inner")
    gdf.to_file(OUT_GPKG, driver="GPKG")
    logger.info(f"GeoPackage → {OUT_GPKG}  ({len(gdf):,} points)")

    flat = df[export_cols]
    flat.to_parquet(OUT_PARQUET, index=False)
    logger.info(f"Parquet → {OUT_PARQUET}")

    # ── 9. Summary ────────────────────────────────────────────────────────
    print("\n── Model Comparison (Spatial Cross-Validation) ─────────────")
    print(f"  {'Model':<20} {'ROC-AUC':>9} {'PR-AUC':>8} {'F1':>8} {'Folds':>7}")
    print(f"  {'-'*52}")
    for name, r in results.items():
        marker = " ◀ best" if name == best_name else ""
        print(f"  {name:<20} {r['roc_auc']:>9.4f} {r['pr_auc']:>8.4f} "
              f"{r['f1']:>8.4f} {r['n_folds']:>7}{marker}")

    print(f"\n── SLFVI Distribution ──────────────────────────────────────")
    print(f"  {'Class':<20} {'Count':>7} {'%':>7}")
    for cls, n in df["vulnerability_class_final"].value_counts().sort_index().items():
        print(f"  {str(cls):<20} {n:>7,} {100*n/len(df):>6.1f}%")

    print(f"\n── Top 10 Drainage Maintenance Priority Segments ───────────")
    print(priority[["highway", "flood_stratum", "latitude", "longitude",
                     "slfvi_final", "maintenance_priority_score"]].head(10).to_string())

    if shap_df is not None:
        print(f"\n── Top 10 Features by SHAP Importance ──────────────────────")
        for _, row in shap_df.head(10).iterrows():
            bar = "█" * int(row["shap_importance"] * 100)
            print(f"  {row['feature']:<45} {row['shap_importance']:.4f}  {bar}")

    print(f"\n── Output Files ─────────────────────────────────────────────")
    for p in [OUT_PARQUET, OUT_GPKG, OUT_MODEL_CMP, OUT_SHAP, OUT_PRIORITY]:
        if p.exists():
            kb = p.stat().st_size / 1024
            print(f"  {p.name:<50} {kb:>8.1f} KB")


if __name__ == "__main__":
    main()
