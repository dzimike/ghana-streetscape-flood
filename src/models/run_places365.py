"""Places365 scene classification for all street-level images.

Uses the WideResNet-18 model bundled with ZenSVI (Zhou et al., 2017 / Ito et al., 2025)
to extract scene category probabilities and scene attribute scores for every image.
Implements image loading with PIL instead of torchvision (incompatible with Python 3.13).

Output:
  data/processed/places365_features.parquet   — per-image (pano_id × heading)
  data/processed/places365_segment.parquet    — per-point aggregated (mean over headings)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import importlib.util, numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from loguru import logger

# ── Resolve ZenSVI bundled assets ─────────────────────────────────────────────
ZENSVI_UTILS = Path(
    importlib.util.find_spec("zensvi").origin
).parent / "cv/classification/utils"

MODEL_PATH    = ZENSVI_UTILS / "wideresnet18_places365.pth.tar"
W_ATTR_PATH   = ZENSVI_UTILS / "W_sceneattribute_wideresnet18.npy"
CATEGORIES_TXT = ZENSVI_UTILS / "categories_places365.txt"
IO_TXT         = ZENSVI_UTILS / "IO_places365.txt"
ATTR_TXT       = ZENSVI_UTILS / "labels_sunattribute.txt"

assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

# ── Load label files ──────────────────────────────────────────────────────────
with open(CATEGORIES_TXT) as f:
    SCENE_CLASSES = tuple(line.strip().split(" ")[0][3:] for line in f)   # 365

with open(IO_TXT) as f:
    LABELS_IO = np.array([int(line.rstrip().split()[-1]) - 1 for line in f])

with open(ATTR_TXT) as f:
    ATTR_LABELS = [line.rstrip() for line in f]                            # 102

W_ATTRIBUTE = np.load(W_ATTR_PATH)   # (102, 512)

# ── Image preprocessing (PIL-based, no torchvision) ──────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0          # H×W×3  [0,1]
    arr = (arr - _MEAN) / _STD                              # normalise
    arr = arr.transpose(2, 0, 1)                            # 3×H×W
    return torch.from_numpy(arr)


# ── Load WideResNet-18 Places365 model ────────────────────────────────────────
def _recursion_change_bn(module: torch.nn.Module) -> torch.nn.Module:
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for _, m in module._modules.items():
            _recursion_change_bn(m)
    return module


def load_model(device: torch.device) -> tuple[torch.nn.Module, list]:
    spec = importlib.util.spec_from_file_location(
        "wideresnet", ZENSVI_UTILS / "wideresnet.py"
    )
    wr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wr)

    model = wr.resnet18(num_classes=365)
    ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state)

    for _, m in model._modules.items():
        _recursion_change_bn(m)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    model.eval()
    model.to(device)

    # Register hooks to capture feature maps
    feature_blobs: list[np.ndarray] = []

    def hook(_, __, output):
        feature_blobs.append(output.squeeze().detach().cpu().numpy())

    model._modules["layer4"].register_forward_hook(hook)
    model._modules["avgpool"].register_forward_hook(hook)

    return model, feature_blobs


# ── Inference ─────────────────────────────────────────────────────────────────
MANIFEST_IN   = ROOT / "data/interim/image_manifest.parquet"
OUT_IMAGE     = ROOT / "data/processed/places365_features.parquet"
OUT_SEGMENT   = ROOT / "data/processed/places365_segment.parquet"
BATCH_SIZE    = 16


def run(sample_only: bool = False) -> None:
    device = torch.device("cpu")
    logger.info(f"Loading Places365 WideResNet-18 from {MODEL_PATH.name} …")
    model, feature_blobs = load_model(device)

    # Softmax weight for scene categories
    params         = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy()
    weight_softmax[weight_softmax < 0] = 0

    manifest = pd.read_parquet(MANIFEST_IN)
    manifest = manifest[manifest["downloaded"].fillna(False)].copy()
    if sample_only:
        manifest = manifest.sample(200, random_state=42)
    logger.info(f"Images to process: {len(manifest):,}")

    rows = []
    paths  = manifest["local_path"].tolist()
    pids   = manifest["pano_id"].tolist()
    heads  = manifest["heading"].tolist()
    pts    = manifest["point_id"].tolist()

    for start in tqdm(range(0, len(paths), BATCH_SIZE), desc="Places365"):
        batch_paths = paths[start: start + BATCH_SIZE]
        batch_pids  = pids[start: start + BATCH_SIZE]
        batch_heads = heads[start: start + BATCH_SIZE]
        batch_pts   = pts[start: start + BATCH_SIZE]

        tensors = []
        valid   = []
        for i, p in enumerate(batch_paths):
            try:
                tensors.append(preprocess(p))
                valid.append(i)
            except Exception as e:
                logger.debug(f"Skip {p}: {e}")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)  # (B, 3, 224, 224)
        feature_blobs.clear()

        with torch.no_grad():
            output = model(batch_tensor)

        probs = F.softmax(output, dim=1).cpu().numpy()   # (B, 365)

        # avgpool features: shape (B, 512) or (512,) for B=1
        if len(valid) == 1:
            feat_avg = feature_blobs[1].reshape(1, -1)   # (1, 512)
        else:
            feat_avg = feature_blobs[1]                  # (B, 512)
        if feat_avg.ndim == 1:
            feat_avg = feat_avg.reshape(1, -1)

        for local_i, global_i in enumerate(valid):
            img_probs  = probs[local_i]                             # (365,)
            attr_resp  = W_ATTRIBUTE.dot(feat_avg[local_i])         # (102,)
            top10_idx  = np.argsort(img_probs)[::-1][:10]
            io_score   = float(np.mean(LABELS_IO[top10_idx]))
            env_type   = "outdoor" if io_score >= 0.5 else "indoor"

            row: dict = {
                "pano_id":  batch_pids[global_i],
                "heading":  batch_heads[global_i],
                "point_id": batch_pts[global_i],
                "env_outdoor": float(io_score),
            }
            for j, cls in enumerate(SCENE_CLASSES):
                row[f"sc_{cls}"] = float(img_probs[j])
            for j, attr in enumerate(ATTR_LABELS):
                row[f"sa_{attr}"] = float(attr_resp[j])
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_IMAGE, index=False)
    logger.info(f"Per-image features → {OUT_IMAGE}  ({len(df):,} rows × {df.shape[1]} cols)")

    # ── Aggregate per road segment (mean over headings) ───────────────────
    feat_cols = [c for c in df.columns if c.startswith(("sc_", "sa_", "env_"))]
    seg = df.groupby("point_id")[feat_cols].mean().reset_index()
    seg.to_parquet(OUT_SEGMENT, index=False)
    logger.info(f"Per-segment features → {OUT_SEGMENT}  ({len(seg):,} rows × {seg.shape[1]} cols)")

    # Quick summary: top 5 scene categories overall
    sc_cols = [c for c in seg.columns if c.startswith("sc_")]
    top5 = seg[sc_cols].mean().sort_values(ascending=False).head(5)
    print("\n── Top 5 scene categories (mean probability) ────────────────")
    for cat, v in top5.items():
        print(f"  {cat.replace('sc_', ''):<35} {v:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true", help="Run on 200 images only (test)")
    args = p.parse_args()
    run(sample_only=args.sample)
