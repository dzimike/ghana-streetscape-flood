"""Ablation study — marginal contribution of streetscape features.

Runs LightGBM under spatial block CV with four feature subsets:
  1. Terrain only
  2. Terrain + exposure
  3. Streetscape only
  4. Terrain + exposure + streetscape (full model, reproduced for comparison)

Outputs
-------
outputs/tables/ablation_results.csv
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Paths ──────────────────────────────────────────────────────────────────────
_SF_V2 = ROOT / "data/processed/streetscape_features_v2.parquet"
_SF_V1 = ROOT / "data/processed/streetscape_features.parquet"
STREETSCAPE = _SF_V2 if _SF_V2.exists() else _SF_V1
GEOSPATIAL  = ROOT / "data/processed/geospatial_features.parquet"
OUT_CSV     = ROOT / "outputs/tables/ablation_results.csv"

TARGET_COL = "any_flood_exposed_100m"

# ── Feature subsets ────────────────────────────────────────────────────────────
TERRAIN_FEATS = [
    "elevation_m", "slope_deg", "local_depression_m",
    "dist_drain_m", "dist_waterway_m",
]

EXPOSURE_FEATS = [
    "building_count_100m", "building_count_250m", "pop_density_per_km2",
]

STREETSCAPE_FEATS = [
    # EfficientNet
    "enet_sensitivity", "enet_vuln_prob",
    "enet_visible_drain_present", "enet_open_gutter_present",
    "enet_blocked_drain_present", "enet_stagnant_water_visible",
    "enet_poor_road_condition", "enet_heavy_impervious_surface",
    "enet_solid_waste_accumulation", "enet_informal_structure_near_drainage",
    "enet_low_lying_street_form", "enet_no_visible_drainage",
    # CLIP
    "clip_blocked_drain_present", "clip_stagnant_water_visible",
    "clip_solid_waste_accumulation", "clip_poor_road_condition",
    "clip_visible_drain_present", "clip_no_visible_drainage",
    "clip_unpaved_shoulder", "clip_low_lying_street_form",
    # Segmentation
    "seg_road", "seg_water", "seg_vegetation", "seg_bare_ground",
    "seg_building", "seg_waste",
    # V1 composite scores
    "streetscape_sensitivity_score", "drain_detection_score",
    "drain_obstruction_score", "waste_risk_score", "water_exposure_score",
    "road_quality_score", "informal_encroachment_score", "low_lying_score",
]

FEATURE_SUBSETS = {
    "Terrain only":              TERRAIN_FEATS,
    "Terrain + exposure":        TERRAIN_FEATS + EXPOSURE_FEATS,
    "Streetscape only":          STREETSCAPE_FEATS,
    "Terrain + exposure + streetscape": TERRAIN_FEATS + EXPOSURE_FEATS + STREETSCAPE_FEATS,
}


# ── Spatial block CV (identical to train_fusion_model.py) ─────────────────────
def spatial_block_cv(df: pd.DataFrame, n_blocks: int = 5) -> list[tuple]:
    df = df.copy()
    lat_bins = pd.qcut(df["latitude"],  q=n_blocks, labels=False, duplicates="drop")
    lon_bins = pd.qcut(df["longitude"], q=n_blocks, labels=False, duplicates="drop")
    df["block"] = lat_bins.astype(str) + "_" + lon_bins.astype(str)
    blocks = df["block"].unique()
    np.random.seed(42)
    np.random.shuffle(blocks)
    fold_size = max(1, len(blocks) // n_blocks)
    folds = []
    for i in range(n_blocks):
        test_blocks = blocks[i * fold_size: (i + 1) * fold_size]
        test_idx  = df[df["block"].isin(test_blocks)].index.tolist()
        train_idx = df[~df["block"].isin(test_blocks)].index.tolist()
        if test_idx and train_idx:
            folds.append((train_idx, test_idx))
    return folds


# ── Standard 5-fold stratified CV ─────────────────────────────────────────────
def standard_cv(df: pd.DataFrame, n_splits: int = 5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y = df[TARGET_COL].fillna(0).astype(int)
    return list(skf.split(df, y))


# ── Evaluate one feature subset ────────────────────────────────────────────────
def evaluate_subset(lgb_cls, X_all: pd.DataFrame, y: pd.Series,
                    features: list[str],
                    std_folds: list, sp_folds: list,
                    subset_name: str) -> dict:
    # Keep only features that exist in X_all
    available = [f for f in features if f in X_all.columns]
    missing = [f for f in features if f not in X_all.columns]
    if missing:
        logger.warning(f"  [{subset_name}] missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if not available:
        logger.error(f"  [{subset_name}] no features available, skipping")
        return {"name": subset_name, "n_features": 0,
                "roc_auc_std": float("nan"), "pr_auc_std": float("nan"),
                "roc_auc_sp":  float("nan"), "pr_auc_sp":  float("nan")}

    X = X_all[available]
    logger.info(f"  [{subset_name}] {len(available)} features")

    def run_folds(fold_list):
        aucs, aps = [], []
        for train_idx, test_idx in fold_list:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if y_te.nunique() < 2:
                continue
            m = lgb_cls(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                class_weight="balanced", random_state=42, verbose=-1
            )
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, proba))
            aps.append(average_precision_score(y_te, proba))
        mean_roc = round(float(np.mean(aucs)), 4) if aucs else float("nan")
        mean_pr  = round(float(np.mean(aps)),  4) if aps  else float("nan")
        sd_roc   = round(float(np.std(aucs, ddof=1)), 4) if len(aucs) > 1 else float("nan")
        sd_pr    = round(float(np.std(aps,  ddof=1)), 4) if len(aps)  > 1 else float("nan")
        return mean_roc, mean_pr, sd_roc, sd_pr

    roc_std, pr_std, sd_roc_std, sd_pr_std = run_folds(std_folds)
    roc_sp,  pr_sp,  sd_roc_sp,  sd_pr_sp  = run_folds(sp_folds)

    logger.info(f"    std CV  ROC={roc_std:.4f}±{sd_roc_std:.4f}  PR={pr_std:.4f}±{sd_pr_std:.4f}")
    logger.info(f"    sp  CV  ROC={roc_sp:.4f}±{sd_roc_sp:.4f}   PR={pr_sp:.4f}±{sd_pr_sp:.4f}")

    return {
        "name":         subset_name,
        "n_features":   len(available),
        "roc_auc_std":  roc_std,  "sd_roc_std": sd_roc_std,
        "pr_auc_std":   pr_std,   "sd_pr_std":  sd_pr_std,
        "roc_auc_sp":   roc_sp,   "sd_roc_sp":  sd_roc_sp,
        "pr_auc_sp":    pr_sp,    "sd_pr_sp":   sd_pr_sp,
    }


def evaluate_subset_with_model(model_cls, model_kwargs: dict,
                               X_all: pd.DataFrame, y: pd.Series,
                               features: list[str],
                               std_folds: list, sp_folds: list) -> tuple:
    """Return (roc_std, pr_std, sd_roc_std, sd_pr_std, roc_sp, pr_sp, sd_roc_sp, sd_pr_sp)."""
    available = [f for f in features if f in X_all.columns]
    if not available:
        return (float("nan"),) * 8
    X = X_all[available]

    def run_folds(fold_list):
        aucs, aps = [], []
        for train_idx, test_idx in fold_list:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if y_te.nunique() < 2:
                continue
            m = model_cls(**model_kwargs)
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, proba))
            aps.append(average_precision_score(y_te, proba))
        mean_roc = round(float(np.mean(aucs)), 4) if aucs else float("nan")
        mean_pr  = round(float(np.mean(aps)),  4) if aps  else float("nan")
        sd_roc   = round(float(np.std(aucs, ddof=1)), 4) if len(aucs) > 1 else float("nan")
        sd_pr    = round(float(np.std(aps,  ddof=1)), 4) if len(aps)  > 1 else float("nan")
        return mean_roc, mean_pr, sd_roc, sd_pr

    std_vals = run_folds(std_folds)
    sp_vals  = run_folds(sp_folds)
    return std_vals + sp_vals


def main() -> None:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    logger.info("Loading feature tables …")
    ss = pd.read_parquet(STREETSCAPE)
    gs = pd.read_parquet(GEOSPATIAL)

    gs_only_cols = [c for c in gs.columns
                    if c not in ss.columns or c in ["point_id", "highway",
                                                    "flood_stratum", "latitude", "longitude"]]
    gs_clean = gs[gs_only_cols]
    df = ss.merge(gs_clean, on=["point_id", "highway", "flood_stratum",
                                 "latitude", "longitude"], how="inner")

    for col in [c[:-2] for c in df.columns if c.endswith("_x")]:
        if f"{col}_x" in df.columns and f"{col}_y" in df.columns:
            df[col] = df[f"{col}_x"].combine_first(df[f"{col}_y"])
            df = df.drop(columns=[f"{col}_x", f"{col}_y"])

    logger.info(f"  Merged: {len(df):,} points")

    y = df[TARGET_COL].fillna(0).astype(int)
    logger.info(f"  Target positive rate: {y.mean():.1%} ({y.sum()} / {len(y)})")

    all_feats = list({f for subset in FEATURE_SUBSETS.values() for f in subset})
    all_feats_present = [f for f in all_feats if f in df.columns]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(df[all_feats_present]),
        columns=all_feats_present,
        index=df.index,
    )

    logger.info("Building CV folds …")
    std_folds = standard_cv(df, n_splits=5)
    sp_folds  = spatial_block_cv(df, n_blocks=5)
    logger.info(f"  Standard folds: {len(std_folds)} | Spatial folds: {len(sp_folds)}")

    full_feats = TERRAIN_FEATS + EXPOSURE_FEATS + STREETSCAPE_FEATS

    # ── Part 1: LightGBM ablation across all four feature subsets ─────────────
    lgb_kwargs = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                      class_weight="balanced", random_state=42, verbose=-1)

    ablation_results = []
    for name, feats in FEATURE_SUBSETS.items():
        logger.info(f"\n[LightGBM] {name}")
        row = evaluate_subset(lgb.LGBMClassifier, X_imp, y,
                              feats, std_folds, sp_folds, name)
        ablation_results.append(row)

    # ── Part 2: RF and XGBoost on full feature set only ───────────────────────
    scale_pos = int((y == 0).sum() / max((y == 1).sum(), 1))

    model_configs = [
        ("Random Forest", RandomForestClassifier,
         dict(n_estimators=300, max_depth=12, min_samples_leaf=5,
              class_weight="balanced", random_state=42, n_jobs=-1)),
        ("XGBoost", xgb.XGBClassifier,
         dict(n_estimators=300, max_depth=6, learning_rate=0.05,
              scale_pos_weight=scale_pos, eval_metric="logloss",
              random_state=42, verbosity=0)),
    ]

    model_comparison = []
    for mname, mcls, mkwargs in model_configs:
        logger.info(f"\n[{mname}] full feature set")
        roc_std, pr_std, sd_roc_std, sd_pr_std, roc_sp, pr_sp, sd_roc_sp, sd_pr_sp = \
            evaluate_subset_with_model(mcls, mkwargs, X_imp, y, full_feats, std_folds, sp_folds)
        logger.info(f"    std CV  ROC={roc_std:.4f}±{sd_roc_std:.4f}  PR={pr_std:.4f}±{sd_pr_std:.4f}")
        logger.info(f"    sp  CV  ROC={roc_sp:.4f}±{sd_roc_sp:.4f}   PR={pr_sp:.4f}±{sd_pr_sp:.4f}")
        model_comparison.append({
            "model": mname,
            "n_features": len([f for f in full_feats if f in X_imp.columns]),
            "roc_auc_std": roc_std, "sd_roc_std": sd_roc_std,
            "pr_auc_std":  pr_std,  "sd_pr_std":  sd_pr_std,
            "roc_auc_sp":  roc_sp,  "sd_roc_sp":  sd_roc_sp,
            "pr_auc_sp":   pr_sp,   "sd_pr_sp":   sd_pr_sp,
        })

    # Add LightGBM full-model row to comparison
    lgb_full = next(r for r in ablation_results if r["name"] == "Terrain + exposure + streetscape")
    model_comparison.append({
        "model": "LightGBM", "n_features": lgb_full["n_features"],
        "roc_auc_std": lgb_full["roc_auc_std"], "sd_roc_std": lgb_full["sd_roc_std"],
        "pr_auc_std":  lgb_full["pr_auc_std"],  "sd_pr_std":  lgb_full["sd_pr_std"],
        "roc_auc_sp":  lgb_full["roc_auc_sp"],  "sd_roc_sp":  lgb_full["sd_roc_sp"],
        "pr_auc_sp":   lgb_full["pr_auc_sp"],   "sd_pr_sp":   lgb_full["sd_pr_sp"],
    })

    # Export
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ablation_results).to_csv(OUT_CSV, index=False)
    OUT_MODEL_CSV = OUT_CSV.parent / "model_comparison_clean.csv"
    pd.DataFrame(model_comparison).to_csv(OUT_MODEL_CSV, index=False)

    # Print ablation table
    print("\n── LightGBM Ablation (Table 2b) ────────────────────────────────────────────")
    print(f"  {'Feature set':<38} {'N':>4} {'ROC(std)':>14} {'PR(std)':>14} "
          f"{'ROC(sp)':>14} {'PR(sp)':>14}")
    print(f"  {'-'*104}")
    for row in ablation_results:
        print(f"  {row['name']:<38} {row['n_features']:>4} "
              f"{row['roc_auc_std']:>6.4f}±{row['sd_roc_std']:.4f} "
              f"{row['pr_auc_std']:>6.4f}±{row['sd_pr_std']:.4f} "
              f"{row['roc_auc_sp']:>6.4f}±{row['sd_roc_sp']:.4f} "
              f"{row['pr_auc_sp']:>6.4f}±{row['sd_pr_sp']:.4f}")

    # Print model comparison table
    print("\n── Model Comparison — clean feature set (Table 2) ─────────────────────────")
    print(f"  {'Model':<20} {'N':>4} {'ROC(std)':>14} {'PR(std)':>14} "
          f"{'ROC(sp)':>14} {'PR(sp)':>14}")
    print(f"  {'-'*86}")
    for row in model_comparison:
        print(f"  {row['model']:<20} {row['n_features']:>4} "
              f"{row['roc_auc_std']:>6.4f}±{row['sd_roc_std']:.4f} "
              f"{row['pr_auc_std']:>6.4f}±{row['sd_pr_std']:.4f} "
              f"{row['roc_auc_sp']:>6.4f}±{row['sd_roc_sp']:.4f} "
              f"{row['pr_auc_sp']:>6.4f}±{row['sd_pr_sp']:.4f}")

    print(f"\n── Outputs → {OUT_CSV}")
    print(f"            {OUT_MODEL_CSV}")


if __name__ == "__main__":
    main()
