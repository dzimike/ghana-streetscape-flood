"""Clean fusion model: CLIP + SegFormer + Places365 + Pixel + Object features.

Pre-trained, off-the-shelf models only — no fine-tuning on local annotations:
  - CLIP zero-shot scene labels          (OpenAI, 2021)
  - SegFormer-B0 semantic segmentation   (NVIDIA, ADE20K pre-trained)
  - WideResNet-18 Places365              (Zhou et al., 2017 / ZenSVI)
  - Pixel statistics                     (PIL-based, no model)
  - YOLOv8n object detection             (COCO pre-trained)

Directly parallels Zhou et al. (2026, Cities) five feature categories:
  semantic | scene (Places365 + diversity) | pixel | object | CLIP/SegFormer

Outputs:
  data/processed/vulnerability_index_v3.parquet
  data/processed/vulnerability_index_v3.gpkg
  outputs/tables/model_comparison_v3.csv
  outputs/tables/shap_feature_importance_v3.csv
  outputs/tables/drainage_maintenance_priority_v3.csv
  models/fusion/lgb_clean.pkl
"""
from __future__ import annotations

import sys, warnings, pickle
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────────
STREETSCAPE   = ROOT / "data/processed/streetscape_features.parquet"        # CLIP + SegFormer
PLACES365     = ROOT / "data/processed/places365_segment.parquet"            # Places365 scene
PIXEL_FEATS   = ROOT / "data/processed/pixel_segment.parquet"               # pixel statistics
OBJECT_FEATS  = ROOT / "data/processed/object_detection_segment.parquet"    # YOLOv8 COCO counts
GEOSPATIAL    = ROOT / "data/processed/geospatial_features.parquet"
SAMPLE_PTS    = ROOT / "data/interim/sample_points.gpkg"

OUT_PARQUET   = ROOT / "data/processed/vulnerability_index_v3.parquet"
OUT_GPKG      = ROOT / "data/processed/vulnerability_index_v3.gpkg"
OUT_MODEL_CMP = ROOT / "outputs/tables/model_comparison_v3.csv"
OUT_SHAP      = ROOT / "outputs/tables/shap_feature_importance_v3.csv"
OUT_PRIORITY  = ROOT / "outputs/tables/drainage_maintenance_priority_v3.csv"
MODEL_STORE   = ROOT / "models/fusion"


def _norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)


# ── SLFVI formula ─────────────────────────────────────────────────────────────
def build_slfvi(df: pd.DataFrame) -> pd.DataFrame:
    """SLFVI = 0.30H + 0.20E + 0.35S + 0.15A  (weights are informed priors)."""

    # H: Hazard (terrain / proximity)
    h_elev  = _norm(-df["elevation_m"].fillna(df["elevation_m"].median()))
    h_depr  = _norm(-df["local_depression_m"].fillna(0))
    h_slope = _norm(-df["slope_deg"].fillna(0))
    h_drain = _norm(-df["dist_drain_m"].fillna(df["dist_drain_m"].median()))
    h_water = _norm(-df["dist_waterway_m"].fillna(df["dist_waterway_m"].median()))
    df["H"] = 0.30*h_elev + 0.25*h_depr + 0.15*h_slope + 0.15*h_drain + 0.15*h_water

    # E: Exposure (buildings / population) — constant across pilot points
    e_bld   = _norm(df["building_count_100m"].fillna(0))
    e_pop   = _norm(df["pop_density_per_km2"].fillna(0))
    e_flood = _norm(df["flood_exposed_count_100m"].fillna(0))
    df["E"] = 0.35*e_flood + 0.35*e_bld + 0.30*e_pop

    # S: Streetscape Sensitivity — CLIP + SegFormer + Places365 composite
    # CLIP drainage/waste signals
    clip_drain  = df.get("drain_detection_score",
                         df.get("clip_visible_drain_present", pd.Series(0.5, index=df.index)))
    clip_waste  = df.get("waste_risk_score",
                         df.get("clip_solid_waste_accumulation", pd.Series(0.5, index=df.index)))
    clip_water  = df.get("water_exposure_score",
                         df.get("clip_stagnant_water_visible", pd.Series(0.5, index=df.index)))
    clip_road   = df.get("road_quality_score",
                         df.get("clip_poor_road_condition", pd.Series(0.5, index=df.index)))
    # SegFormer surface fractions
    seg_imperf  = df.get("seg_road", pd.Series(0.0, index=df.index)) + \
                  df.get("seg_building", pd.Series(0.0, index=df.index))
    # Places365 outdoor scene score (outdoor = more street-level vulnerability signal)
    p365_out    = df.get("env_outdoor", pd.Series(0.5, index=df.index))

    df["S"] = _norm(
        0.35 * _norm(clip_drain.fillna(0.5))
        + 0.25 * _norm(clip_waste.fillna(0.5))
        + 0.20 * _norm(clip_water.fillna(0.5))
        + 0.10 * _norm(seg_imperf.fillna(0))
        + 0.10 * _norm(p365_out.fillna(0.5))
    )

    # A: Adaptive capacity (inverse — 0 = best access, 1 = worst)
    if "adaptive_capacity_raw_score" in df.columns:
        df["A"] = 1.0 - df["adaptive_capacity_raw_score"].fillna(
            df["adaptive_capacity_raw_score"].median()
        )
    else:
        df["A"] = _norm(clip_road.fillna(0.5))

    df["slfvi"] = _norm(0.30*df["H"] + 0.20*df["E"] + 0.35*df["S"] + 0.15*df["A"])
    bins   = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
    labels = ["very_low", "low", "moderate", "high", "very_high"]
    df["vulnerability_class"] = pd.cut(df["slfvi"], bins=bins, labels=labels)
    return df


# ── Feature matrix (clean — no EfficientNet) ──────────────────────────────────
# Core terrain + exposure + CLIP + SegFormer
BASE_FEATURES = [
    # Terrain / Hazard
    "elevation_m", "slope_deg", "local_depression_m",
    "dist_drain_m", "dist_waterway_m",
    # Exposure
    "building_count_100m", "building_count_250m", "pop_density_per_km2",
    # CLIP zero-shot signals (pre-trained OpenAI CLIP — no local training)
    "clip_blocked_drain_present", "clip_stagnant_water_visible",
    "clip_solid_waste_accumulation", "clip_poor_road_condition",
    "clip_visible_drain_present", "clip_no_visible_drainage",
    "clip_unpaved_shoulder", "clip_low_lying_street_form",
    "clip_informal_structure_near_drainage", "clip_heavy_impervious_surface",
    "clip_open_gutter_present", "clip_visible_waterway_or_stream",
    "clip_roadside_erosion", "clip_pedestrian_exposure",
    "clip_culvert_or_bridge_visible",
    # Composite CLIP scores
    "drain_detection_score", "drain_obstruction_score",
    "waste_risk_score", "water_exposure_score", "road_quality_score",
    "informal_encroachment_score", "low_lying_score",
    # SegFormer segmentation fractions (pre-trained ADE20K — no local training)
    "seg_road", "seg_water", "seg_vegetation", "seg_bare_ground",
    "seg_building", "seg_waste", "seg_sidewalk", "seg_kiosk",
    # SLFVI components
    "H", "E", "S", "A",
]

# Places365 features added if available
TOP_SCENE_CATS = [
    # Outdoor/street scene categories most relevant to flood vulnerability
    "street", "road", "highway", "alley", "slum", "ghetto",
    "canal_natural", "swamp", "marsh", "flood", "river",
    "yard", "parking_lot", "construction_site",
    "residential_neighborhood", "market_outdoor", "bazaar_outdoor",
]
TOP_SCENE_ATTRS = [
    "natural_light", "open_area", "man-made", "transporting",
    "vegetation", "water", "dirty", "unpleasant", "rugged",
]

TARGET_COL = "any_flood_exposed_100m"

# ── Scene diversity indices (from Places365 distribution) ────────────────────
def add_scene_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Add 6 scene diversity indices computed from Places365 category probabilities.

    Parallels the 6 scene diversity variables in Zhou et al. (2026, Cities):
      scene_shannon     — Shannon entropy across 365 category probs (scene richness)
      scene_simpson     — Simpson diversity index (1 - sum(p²))
      scene_top1_prob   — probability of the single dominant scene
      scene_top5_share  — cumulative probability of top-5 scene categories
      scene_outdoor_frac — fraction of probability mass in outdoor categories
      scene_entropy_norm — scene_shannon normalised by log(365)
    """
    sc_cols = [c for c in df.columns if c.startswith("sc_")]
    if not sc_cols:
        return df

    probs = df[sc_cols].fillna(0).values.astype(np.float64)
    # Normalise rows to sum to 1 (they should already, but guard against NaN fill)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    p = probs / row_sums

    # Shannon entropy H = -sum(p * log(p))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    shannon = -(p * log_p).sum(axis=1)

    # Simpson = 1 - sum(p^2)
    simpson = 1.0 - (p ** 2).sum(axis=1)

    # Top-1 probability
    top1 = p.max(axis=1)

    # Top-5 cumulative probability
    top5 = np.sort(p, axis=1)[:, -5:].sum(axis=1)

    # Outdoor fraction: ZenSVI IO_places365.txt marks outdoor=1
    # We use env_outdoor (already computed in run_places365.py) as the scalar proxy;
    # here we compute the fraction of probability mass in columns labelled sc_*
    # that belong to known outdoor scene categories.
    outdoor_keywords = {
        "street", "road", "highway", "alley", "sidewalk", "path", "yard",
        "parking", "plaza", "market", "canal", "river", "lake", "swamp",
        "marsh", "flood", "construction", "residential", "urban",
    }
    outdoor_mask = np.array([
        any(kw in c.replace("sc_", "") for kw in outdoor_keywords)
        for c in sc_cols
    ])
    outdoor_frac = p[:, outdoor_mask].sum(axis=1)

    # Normalised entropy
    entropy_norm = shannon / np.log(max(len(sc_cols), 2))

    df["scene_shannon"]      = shannon
    df["scene_simpson"]      = simpson
    df["scene_top1_prob"]    = top1
    df["scene_top5_share"]   = top5
    df["scene_outdoor_frac"] = outdoor_frac
    df["scene_entropy_norm"] = entropy_norm
    return df


SCENE_DIVERSITY_COLS = [
    "scene_shannon", "scene_simpson", "scene_top1_prob",
    "scene_top5_share", "scene_outdoor_frac", "scene_entropy_norm",
]


def spatial_block_cv(df: pd.DataFrame, n_blocks: int = 5) -> list[tuple]:
    df = df.copy()
    lat_q = pd.qcut(df["latitude"],  q=n_blocks, labels=False, duplicates="drop")
    lon_q = pd.qcut(df["longitude"], q=n_blocks, labels=False, duplicates="drop")
    df["block"] = lat_q.astype(str) + "_" + lon_q.astype(str)
    blocks = df["block"].unique()
    np.random.seed(42)
    np.random.shuffle(blocks)
    fold_sz = max(1, len(blocks) // n_blocks)
    folds = []
    for i in range(n_blocks):
        test_bl  = blocks[i * fold_sz: (i + 1) * fold_sz]
        test_idx  = df[df["block"].isin(test_bl)].index.tolist()
        train_idx = df[~df["block"].isin(test_bl)].index.tolist()
        if test_idx and train_idx:
            folds.append((train_idx, test_idx))
    return folds


def train_evaluate(model, X: pd.DataFrame, y: pd.Series, folds: list) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    aucs, aps, f1s = [], [], []
    for tr_idx, te_idx in folds:
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        if yte.nunique() < 2:
            continue
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        pred = model.predict(Xte)
        aucs.append(roc_auc_score(yte, prob))
        aps.append(average_precision_score(yte, prob))
        f1s.append(f1_score(yte, pred, zero_division=0))
    return {
        "roc_auc": round(np.mean(aucs), 4) if aucs else np.nan,
        "pr_auc":  round(np.mean(aps),  4) if aps  else np.nan,
        "f1":      round(np.mean(f1s),  4) if f1s  else np.nan,
        "n_folds": len(aucs),
    }


def main() -> None:
    from sklearn.impute import SimpleImputer
    import geopandas as gpd

    # ── 1. Load and merge ─────────────────────────────────────────────────
    logger.info("Loading feature tables …")
    ss = pd.read_parquet(STREETSCAPE)
    gs = pd.read_parquet(GEOSPATIAL)

    gs_only = [c for c in gs.columns
               if c not in ss.columns or c in ["point_id","highway","flood_stratum","latitude","longitude"]]
    df = ss.merge(gs[gs_only], on=["point_id","highway","flood_stratum","latitude","longitude"],
                  how="inner")

    # Merge Places365 features if available
    if PLACES365.exists():
        p365 = pd.read_parquet(PLACES365)
        logger.info(f"  Places365: {p365.shape[1]-1} scene features for {len(p365):,} points")
        df = df.merge(p365, on="point_id", how="left")
    else:
        logger.warning("Places365 features not found — run run_places365.py first")

    # Merge pixel features if available
    if PIXEL_FEATS.exists():
        pix = pd.read_parquet(PIXEL_FEATS)
        logger.info(f"  Pixel: {pix.shape[1]-1} features for {len(pix):,} points")
        df = df.merge(pix, on="point_id", how="left")
    else:
        logger.warning("Pixel features not found — run run_pixel_features.py first")

    # Merge object detection features if available
    if OBJECT_FEATS.exists():
        obj = pd.read_parquet(OBJECT_FEATS)
        logger.info(f"  Object: {obj.shape[1]-1} COCO class counts for {len(obj):,} points")
        df = df.merge(obj, on="point_id", how="left")
    else:
        logger.warning("Object detection features not found — run run_object_detection.py first")

    # Resolve _x/_y columns
    for col in [c[:-2] for c in df.columns if c.endswith("_x")]:
        if f"{col}_x" in df.columns and f"{col}_y" in df.columns:
            df[col] = df[f"{col}_x"].combine_first(df[f"{col}_y"])
            df = df.drop(columns=[f"{col}_x", f"{col}_y"])

    logger.info(f"  Merged: {len(df):,} points")

    # ── 1b. Scene diversity indices (derived from Places365 probs) ───────
    df = add_scene_diversity(df)

    # ── 2. Build SLFVI ────────────────────────────────────────────────────
    df = build_slfvi(df)

    # ── 3. Feature matrix ─────────────────────────────────────────────────
    feature_cols = BASE_FEATURES.copy()

    # Add available Places365 scene and attribute features
    sc_cols = [c for c in df.columns if c.startswith("sc_")]
    sa_cols = [c for c in df.columns if c.startswith("sa_")]
    if sc_cols:
        # Use top N most variable scene categories to limit dimensionality
        variance = df[sc_cols].var()
        top_sc   = variance.nlargest(50).index.tolist()
        feature_cols += top_sc
        logger.info(f"  Added {len(top_sc)} top-variance scene categories")
    if sa_cols:
        feature_cols += sa_cols[:30]   # top 30 scene attributes by list order
        logger.info(f"  Added {min(30, len(sa_cols))} scene attributes")
    if "env_outdoor" in df.columns:
        feature_cols.append("env_outdoor")

    # Scene diversity indices (derived from Places365)
    div_cols = [c for c in SCENE_DIVERSITY_COLS if c in df.columns]
    if div_cols:
        feature_cols += div_cols
        logger.info(f"  Added {len(div_cols)} scene diversity indices")

    # Pixel features (9 low-level statistics)
    pix_cols = [c for c in [
        "mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v",
        "contrast", "green_ratio", "sky_ratio",
    ] if c in df.columns]
    if pix_cols:
        feature_cols += pix_cols
        logger.info(f"  Added {len(pix_cols)} pixel features")

    # Object detection: top 32 most variable COCO classes (parallels Cities paper)
    obj_cols_all = [c for c in df.columns if c.startswith("obj_")]
    if obj_cols_all:
        variance = df[obj_cols_all].var()
        top_obj  = variance.nlargest(32).index.tolist()
        feature_cols += top_obj
        logger.info(f"  Added {len(top_obj)} object detection features (top-32 by variance)")

    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing[:5]:
        logger.warning(f"Missing features (first 5): {missing[:5]}")

    X_raw = df[available].copy()
    y     = df[TARGET_COL].fillna(0).astype(int)

    imp = SimpleImputer(strategy="median")
    X   = pd.DataFrame(imp.fit_transform(X_raw), columns=available, index=df.index)

    logger.info(f"  Feature matrix: {len(available)} cols | Positive rate: {y.mean():.1%}")

    # ── 4. Spatial CV + train ─────────────────────────────────────────────
    folds   = spatial_block_cv(df)
    results = {}
    models  = {}

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    results["RandomForest"] = train_evaluate(rf, X, y, folds)
    rf.fit(X, y)
    models["RandomForest"] = rf
    logger.info(f"  RF  ROC-AUC={results['RandomForest']['roc_auc']}")

    try:
        import lightgbm as lgb
        lgb_m = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   class_weight="balanced", random_state=42, verbose=-1)
        results["LightGBM"] = train_evaluate(lgb_m, X, y, folds)
        lgb_m.fit(X, y)
        models["LightGBM"] = lgb_m
        logger.info(f"  LGB ROC-AUC={results['LightGBM']['roc_auc']}")
    except Exception as e:
        logger.warning(f"LightGBM skipped: {e}")

    best_name  = max(results, key=lambda k: results[k]["roc_auc"] or 0)
    best_model = models[best_name]
    logger.info(f"  Best model: {best_name}")

    MODEL_STORE.mkdir(parents=True, exist_ok=True)
    with open(MODEL_STORE / "lgb_clean.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_STORE / "feature_cols_clean.pkl", "wb") as f:
        pickle.dump(available, f)

    # ── 5. Final scores ───────────────────────────────────────────────────
    df["ml_flood_prob"] = best_model.predict_proba(X)[:, 1]
    df["slfvi_final"]   = _norm(0.70*df["slfvi"] + 0.30*_norm(df["ml_flood_prob"]))
    df["vulnerability_class_final"] = pd.cut(
        df["slfvi_final"], bins=[-0.001,0.20,0.40,0.60,0.80,1.001],
        labels=["very_low","low","moderate","high","very_high"]
    )

    # ── 6. SHAP ───────────────────────────────────────────────────────────
    shap_df = None
    try:
        import shap
        explainer = shap.TreeExplainer(best_model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        shap_df = pd.DataFrame({
            "feature": available,
            "shap_importance": np.abs(sv).mean(axis=0),
        }).sort_values("shap_importance", ascending=False)
        OUT_SHAP.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(OUT_SHAP, index=False)
        logger.info(f"SHAP → {OUT_SHAP}")
    except Exception as e:
        logger.warning(f"SHAP skipped: {e}")

    # ── 7. Export ─────────────────────────────────────────────────────────
    cmp_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    OUT_MODEL_CMP.parent.mkdir(parents=True, exist_ok=True)
    cmp_df.to_csv(OUT_MODEL_CMP, index=False)

    obstruction = df.get("drain_obstruction_score",
                         df.get("clip_blocked_drain_present", pd.Series(0, index=df.index)))
    waste_risk  = df.get("waste_risk_score",
                         df.get("clip_solid_waste_accumulation", pd.Series(0, index=df.index)))
    df["drain_obstruction_score"] = obstruction.fillna(0)
    df["waste_risk_score"]        = waste_risk.fillna(0)
    df["maintenance_priority_score"] = (
        0.50*df["slfvi_final"] + 0.30*df["drain_obstruction_score"] + 0.20*df["waste_risk_score"]
    )

    priority = (
        df[["point_id","highway","flood_stratum","latitude","longitude",
            "slfvi_final","drain_obstruction_score","waste_risk_score",
            "maintenance_priority_score","vulnerability_class_final"]]
        .sort_values("maintenance_priority_score", ascending=False)
        .head(100).reset_index(drop=True)
    )
    priority.index += 1
    priority.index.name = "rank"
    priority.to_csv(OUT_PRIORITY)

    export_cols = [c for c in [
        "point_id","highway","flood_stratum","latitude","longitude",
        "H","E","S","A","slfvi","slfvi_final",
        "vulnerability_class","vulnerability_class_final",
        "ml_flood_prob","maintenance_priority_score",
        "drain_obstruction_score","waste_risk_score",
        "drain_detection_score","streetscape_sensitivity_score",
        "elevation_m","slope_deg","local_depression_m",
        "building_count_100m","flood_exposed_count_100m","pop_density_per_km2",
    ] if c in df.columns]

    df[export_cols].to_parquet(OUT_PARQUET, index=False)

    try:
        sp  = gpd.read_file(SAMPLE_PTS)[["point_id","geometry"]]
        gdf = sp.merge(df[export_cols], on="point_id", how="inner")
        gdf.to_file(OUT_GPKG, driver="GPKG")
        logger.info(f"GeoPackage → {OUT_GPKG}")
    except Exception as e:
        logger.warning(f"GeoPackage skipped: {e}")

    # ── 8. Print summary ──────────────────────────────────────────────────
    print("\n── Model Comparison (Spatial Block CV, clean features) ──────")
    print(f"  {'Model':<20} {'ROC-AUC':>9} {'PR-AUC':>8} {'F1':>8}")
    for name, r in results.items():
        mark = " ◀" if name == best_name else ""
        print(f"  {name:<20} {r['roc_auc']:>9.4f} {r['pr_auc']:>8.4f} {r['f1']:>8.4f}{mark}")

    print(f"\n── SLFVI class distribution ─────────────────────────────────")
    for cls, n in df["vulnerability_class_final"].value_counts().sort_index().items():
        print(f"  {str(cls):<20} {n:>6,}  ({100*n/len(df):>5.1f}%)")

    if shap_df is not None:
        print(f"\n── Top 15 Features by SHAP ──────────────────────────────────")
        for _, row in shap_df.head(15).iterrows():
            bar = "█" * int(row["shap_importance"] * 100)
            print(f"  {row['feature']:<45} {row['shap_importance']:.4f}  {bar}")

    print(f"\n── Outputs ──────────────────────────────────────────────────")
    for p in [OUT_PARQUET, OUT_GPKG, OUT_MODEL_CMP, OUT_SHAP, OUT_PRIORITY]:
        if p.exists():
            print(f"  {p.name:<50} {p.stat().st_size/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
