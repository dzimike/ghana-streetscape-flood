"""Task 6: Train image-level flood vulnerability classifier.

Two tasks trained jointly on EfficientNet-B0:
  1. Binary vulnerability (low vs moderate+high)  — most actionable signal
  2. Multi-label (15 flood indicator labels)       — per-label detection

Uses MPS (Apple Silicon) when available.
Saves best model checkpoint + full metrics to outputs/tables/.

Usage
-----
  python src/models/train_cv_classifier.py            # full training
  python src/models/train_cv_classifier.py --epochs 3 --batch 8  # quick test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Config ────────────────────────────────────────────────────────────────────
ANNOTATIONS  = ROOT / "data/interim/openai_annotation_summary.csv"
GOLD_LABELS  = ROOT / "data/interim/gold_labels.csv"
SAMPLE_CSV   = ROOT / "data/interim/annotation_sample.csv"
MODEL_DIR    = ROOT / "models/classification"
OUT_METRICS  = ROOT / "outputs/tables/cv_model_metrics.csv"
OUT_CONF_MAT = ROOT / "outputs/tables/cv_confusion_matrix.csv"
OUT_LABEL_METRICS = ROOT / "outputs/tables/cv_label_metrics.csv"
CHECKPOINT   = MODEL_DIR / "efficientnet_b0_flood.pth"
CHECKPOINT_GOLD = MODEL_DIR / "efficientnet_b0_flood_gold.pth"

IMAGE_SIZE   = 224
LABEL_COLS   = [
    "visible_drain_present", "open_gutter_present", "blocked_drain_present",
    "stagnant_water_visible", "poor_road_condition", "heavy_impervious_surface",
    "unpaved_shoulder", "informal_structure_near_drainage", "solid_waste_accumulation",
    "visible_waterway_or_stream", "low_lying_street_form", "roadside_erosion",
    "pedestrian_exposure", "culvert_or_bridge_visible", "no_visible_drainage",
]
N_LABELS = len(LABEL_COLS)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class FloodDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["local_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        vuln_label  = int(row["vuln_binary"])
        label_vec   = torch.tensor(row[LABEL_COLS].values.astype(np.float32))
        return img, vuln_label, label_vec


# ── Model ─────────────────────────────────────────────────────────────────────
class FloodClassifier(nn.Module):
    def __init__(self, n_labels: int):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        n_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.dropout  = nn.Dropout(0.3)
        # Head 1: binary vulnerability (low=0 vs moderate+high=1)
        self.vuln_head  = nn.Linear(n_features, 2)
        # Head 2: multi-label (15 flood indicators)
        self.label_head = nn.Linear(n_features, n_labels)

    def forward(self, x):
        feat       = self.dropout(self.backbone(x))
        vuln_logit = self.vuln_head(feat)
        label_logit = self.label_head(feat)
        return vuln_logit, label_logit


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    ce_loss  = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0.0

    for imgs, vuln_labels, label_vecs in loader:
        imgs        = imgs.to(device)
        vuln_labels = vuln_labels.to(device)
        label_vecs  = label_vecs.to(device)

        optimizer.zero_grad()
        vuln_logit, label_logit = model(imgs)

        loss = ce_loss(vuln_logit, vuln_labels) + 0.5 * bce_loss(label_logit, label_vecs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_vuln_true, all_vuln_pred, all_vuln_prob = [], [], []
    all_label_true, all_label_prob = [], []

    for imgs, vuln_labels, label_vecs in loader:
        imgs = imgs.to(device)
        vuln_logit, label_logit = model(imgs)

        vuln_prob  = torch.softmax(vuln_logit, dim=1)[:, 1].cpu().numpy()
        vuln_pred  = vuln_logit.argmax(dim=1).cpu().numpy()
        label_prob = torch.sigmoid(label_logit).cpu().numpy()

        all_vuln_true.extend(vuln_labels.numpy())
        all_vuln_pred.extend(vuln_pred)
        all_vuln_prob.extend(vuln_prob)
        all_label_true.append(label_vecs.numpy())
        all_label_prob.append(label_prob)

    vuln_true  = np.array(all_vuln_true)
    vuln_pred  = np.array(all_vuln_pred)
    vuln_prob  = np.array(all_vuln_prob)
    label_true = np.vstack(all_label_true)
    label_prob = np.vstack(all_label_prob)

    f1  = f1_score(vuln_true, vuln_pred, average="binary", zero_division=0)
    auc = roc_auc_score(vuln_true, vuln_prob) if len(np.unique(vuln_true)) > 1 else 0.5
    return f1, auc, vuln_true, vuln_pred, vuln_prob, label_true, label_prob


# ── Main ──────────────────────────────────────────────────────────────────────
def main(epochs: int = 15, batch: int = 16, lr: float = 3e-4, use_gold: bool = False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = CHECKPOINT_GOLD if use_gold else CHECKPOINT

    # ── Load and merge data ───────────────────────────────────────────────────
    sample = pd.read_csv(SAMPLE_CSV)

    if use_gold:
        logger.info("Loading gold labels (human-reviewed) …")
        gold = pd.read_csv(GOLD_LABELS)
        # Match image_file → local_path via sample CSV
        sample["image_file"] = sample["local_path"].apply(lambda p: Path(p).name)
        df = gold.merge(sample[["image_file", "local_path"]], on="image_file")
        logger.info(f"  Human labels: {(gold.label_source=='human').sum()} | "
                    f"AI labels: {(gold.label_source=='ai_gpt4o_mini').sum()}")
    else:
        logger.info("Loading AI labels (GPT-4o-mini) …")
        ann = pd.read_csv(ANNOTATIONS)
        df  = ann.merge(sample[["annotation_id", "local_path"]], on="annotation_id")

    # Binary vulnerability: moderate + high + uncertain → 1, low → 0
    df["vuln_binary"] = (df["vulnerability_class"] != "low_flood_vulnerability").astype(int)
    label_src = "gold" if use_gold else "AI"
    logger.info(f"Dataset [{label_src}]: {len(df)} images | "
                f"positive (non-low): {df['vuln_binary'].sum()} ({df['vuln_binary'].mean()*100:.1f}%)")

    # ── Train / val / test split (stratified) ────────────────────────────────
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["vuln_binary"],
                                         random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["vuln_binary"],
                                         random_state=42)
    logger.info(f"Split → train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    train_ds = FloodDataset(train_df, TRAIN_TRANSFORM)
    val_ds   = FloodDataset(val_df,   VAL_TRANSFORM)
    test_ds  = FloodDataset(test_df,  VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=0)

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Device: {device}")

    # ── Class weights for imbalanced multi-label ──────────────────────────────
    label_counts = train_df[LABEL_COLS].sum()
    pos_weight   = torch.tensor(
        ((len(train_df) - label_counts) / (label_counts + 1)).values.astype(np.float32)
    ).to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = FloodClassifier(N_LABELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training ──────────────────────────────────────────────────────────────
    best_f1    = 0.0
    history    = []

    logger.info(f"Training EfficientNet-B0 for {epochs} epochs …")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_f1, val_auc, *_ = eval_epoch(model, val_loader, device)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_f1": val_f1, "val_auc": val_auc})
        logger.info(f"  Epoch {epoch:02d}/{epochs} | loss {train_loss:.4f} | "
                    f"val F1 {val_f1:.3f} | val AUC {val_auc:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint)
            logger.info(f"    ✓ Saved best model (F1={best_f1:.3f})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("Loading best checkpoint for test evaluation …")
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    (test_f1, test_auc,
     vuln_true, vuln_pred, vuln_prob,
     label_true, label_prob) = eval_epoch(model, test_loader, device)

    precision = precision_score(vuln_true, vuln_pred, zero_division=0)
    recall    = recall_score(vuln_true, vuln_pred, zero_division=0)

    print("\n── Vulnerability Classifier (Test Set) ─────────────────────")
    print(f"  ROC-AUC:   {test_auc:.3f}")
    print(f"  F1:        {test_f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print()
    print(classification_report(vuln_true, vuln_pred,
          target_names=["low", "moderate+high"], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(vuln_true, vuln_pred)
    print("  Confusion matrix (rows=true, cols=pred):")
    cm_df = pd.DataFrame(cm, index=["true_low", "true_mod+high"],
                         columns=["pred_low", "pred_mod+high"])
    print(cm_df.to_string())
    # Per-label metrics
    label_pred = (label_prob > 0.5).astype(int)
    label_rows = []
    for i, lbl in enumerate(LABEL_COLS):
        n_pos = int(label_true[:, i].sum())
        if n_pos == 0:
            ap = pr = rc = f1 = 0.0
        else:
            ap = average_precision_score(label_true[:, i], label_prob[:, i])
            pr = precision_score(label_true[:, i], label_pred[:, i], zero_division=0)
            rc = recall_score(label_true[:, i], label_pred[:, i], zero_division=0)
            f1 = f1_score(label_true[:, i], label_pred[:, i], zero_division=0)
        label_rows.append({
            "label": lbl, "n_positive": n_pos,
            "avg_precision": round(ap, 3),
            "precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3),
        })
    label_df = pd.DataFrame(label_rows).sort_values("avg_precision", ascending=False)

    print("\n── Per-Label Metrics (Test Set) ────────────────────────────")
    print(f"  {'Label':<45} {'N+':>4} {'AP':>6} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*73}")
    for _, r in label_df.iterrows():
        print(f"  {r['label']:<45} {r['n_positive']:>4} "
              f"{r['avg_precision']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f}")
    # Summary metrics CSV — gold run saves alongside baseline for comparison
    suffix = "_gold" if use_gold else ""
    out_metrics   = OUT_METRICS.parent / (OUT_METRICS.stem + suffix + OUT_METRICS.suffix)
    out_conf_mat  = OUT_CONF_MAT.parent / (OUT_CONF_MAT.stem + suffix + OUT_CONF_MAT.suffix)
    out_lbl_mets  = OUT_LABEL_METRICS.parent / (OUT_LABEL_METRICS.stem + suffix + OUT_LABEL_METRICS.suffix)

    baseline_auc = 0.757
    baseline_f1  = 0.594

    metrics_df = pd.DataFrame([{
        "model":          "EfficientNet-B0",
        "label_source":   "gold" if use_gold else "ai_gpt4o_mini",
        "task":           "binary_vulnerability",
        "test_roc_auc":   round(test_auc, 3),
        "test_f1":        round(test_f1, 3),
        "test_precision": round(precision, 3),
        "test_recall":    round(recall, 3),
        "train_size":     len(train_df),
        "val_size":       len(val_df),
        "test_size":      len(test_df),
        "epochs":         epochs,
        "best_val_f1":    round(best_f1, 3),
    }])
    metrics_df.to_csv(out_metrics, index=False)
    cm_df.to_csv(out_conf_mat)
    label_df.to_csv(out_lbl_mets, index=False)

    if use_gold:
        print(f"\n── Comparison: Gold vs Baseline (AI labels) ────────────────")
        print(f"  {'Metric':<20} {'Baseline':>10} {'Gold':>10} {'Δ':>8}")
        print(f"  {'-'*50}")
        print(f"  {'ROC-AUC':<20} {baseline_auc:>10.3f} {test_auc:>10.3f} {test_auc-baseline_auc:>+8.3f}")
        print(f"  {'F1':<20} {baseline_f1:>10.3f} {test_f1:>10.3f} {test_f1-baseline_f1:>+8.3f}")

    print(f"\n── Output Files ────────────────────────────────────────────")
    for p in [checkpoint, out_metrics, out_conf_mat, out_lbl_mets]:
        if p.exists():
            kb = p.stat().st_size / 1024
            tag = f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.1f} KB"
            print(f"  {p.relative_to(ROOT)!s:<55} {tag}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=15)
    p.add_argument("--batch",     type=int,   default=16)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--use-gold",  action="store_true",
                   help="Train on gold labels (human + AI) instead of AI-only labels")
    args = p.parse_args()
    main(epochs=args.epochs, batch=args.batch, lr=args.lr, use_gold=args.use_gold)
