"""
Region-holdout: does the model generalize to space it never trained on?

The domain is split into two spatially disjoint regions. Models are trained only
on region A and evaluated on both region A and the held-out region B. This is a
simple spatial-transfer test, not a final hydrologic validation.

v1.2 additions:
  - reuses cached array.npy/norm.npz when available, avoiding unnecessary reloads
  - applies invalid-cell masks separately to each region

    python src/region_holdout.py --config config_demo.yaml
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data import build_array, chronological_splits, standardize, SeqDataset
from models import build_model

DROUGHT_THR = -0.8


def onset_f1(pred0, actual0, prev0, mask=None):
    if mask is not None:
        pred0 = pred0[..., mask]
        actual0 = actual0[..., mask]
        prev0 = prev0[..., mask]
    pred_on = (prev0 >= DROUGHT_THR) & (pred0 < DROUGHT_THR)
    true_on = (prev0 >= DROUGHT_THR) & (actual0 < DROUGHT_THR)
    tp = (pred_on & true_on).sum()
    fp = (pred_on & ~true_on).sum()
    fn = (~pred_on & true_on).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))


def masked_mse(pred, target, mask=None):
    err = (pred - target) ** 2
    if mask is None:
        return err.mean()
    m = torch.as_tensor(mask, device=pred.device, dtype=pred.dtype).view(1, 1, *mask.shape)
    denom = m.sum() * pred.shape[0] * pred.shape[1]
    return (err * m).sum() / denom.clamp_min(1.0)


def infer_valid_mask(arr):
    mask = np.nanstd(arr[:, 0], axis=0) > 1e-6
    if mask.sum() == 0:
        mask = np.ones(arr.shape[-2:], dtype=bool)
    return mask.astype(bool)


def load_or_build(cfg, out):
    arr_path = os.path.join(out, "array.npy")
    nz_path = os.path.join(out, "norm.npz")
    if os.path.exists(arr_path) and os.path.exists(nz_path):
        arr = np.load(arr_path)
        nz = np.load(nz_path, allow_pickle=True)
        return arr, int(nz["L"]), nz["tr"], nz["va"], nz["te"], nz["valid_mask"] if "valid_mask" in nz.files else infer_valid_mask(arr)

    arr_raw, _ = build_array(cfg)
    L = cfg["model"]["seq_len"]
    tr, va, te = chronological_splits(arr_raw.shape[0], L,
                                      cfg["data"].get("frac_train", 0.7),
                                      cfg["data"].get("frac_val", 0.15))
    mask = infer_valid_mask(arr_raw)
    arr_std, _, _ = standardize(arr_raw, tr)
    return arr_std, L, tr, va, te, mask


def train_on_region(arr, mask, L, tr, cfg, device):
    models = {}
    for name in ["convlstm", "pixel_lstm"]:
        m = build_model(name, arr.shape[1], cfg["model"]).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=cfg["train"]["lr"])
        loader = DataLoader(SeqDataset(arr, L, tr), batch_size=cfg["train"]["batch"], shuffle=True)
        for ep in range(cfg["train"].get("region_epochs", cfg["train"]["epochs"])):
            m.train()
            total, n = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = masked_mse(m(x), y, mask)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * x.size(0)
                n += x.size(0)
            print(f"[{name}] region epoch {ep:02d} loss {total / max(n, 1):.4f}")
        m.eval()
        models[name] = m
    return models


def eval_region(model, arr, mask, L, times, device):
    X = np.stack([arr[t - L:t] for t in times])
    Y = np.stack([arr[t] for t in times])
    PREV = np.stack([arr[t - 1] for t in times])
    with torch.no_grad():
        P = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    p0, y0, prev0 = P[:, 0], Y[:, 0], PREV[:, 0]
    if mask is not None:
        p0f, y0f, prev0f = p0[..., mask], y0[..., mask], prev0[..., mask]
    else:
        p0f, y0f, prev0f = p0, y0, prev0
    mse_persist = ((prev0f - y0f) ** 2).mean() + 1e-12
    skill = float(1 - ((p0f - y0f) ** 2).mean() / mse_persist)
    return skill, onset_f1(p0, y0, prev0, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))

    arr_std, L, tr, va, te, valid_mask = load_or_build(cfg, out)

    W = arr_std.shape[-1]
    cut = W // 2
    region_A = arr_std[..., :cut].copy()
    region_B = arr_std[..., cut:].copy()
    mask_A = valid_mask[..., :cut]
    mask_B = valid_mask[..., cut:]

    models = train_on_region(region_A, mask_A, L, tr, cfg, device)

    res = {}
    for name, m in models.items():
        sA, fA = eval_region(m, region_A, mask_A, L, te, device)
        sB, fB = eval_region(m, region_B, mask_B, L, te, device)
        res[name] = {"in_region": {"skill": sA, "onset_F1": fA},
                     "held_out_region": {"skill": sB, "onset_F1": fB},
                     "skill_retention": float(sB / sA) if abs(sA) > 1e-9 else None}
    json.dump(res, open(os.path.join(out, "region_holdout.json"), "w"), indent=2)

    names = ["convlstm", "pixel_lstm"]
    labels = {"convlstm": "ConvLSTM", "pixel_lstm": "Pixel LSTM"}
    x = np.arange(len(names))
    bw = 0.36
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for j, metric, title in [(0, "skill", "Skill vs persistence"), (1, "onset_F1", "Drought-onset F1")]:
        inr = [res[n]["in_region"][metric] for n in names]
        out_ = [res[n]["held_out_region"][metric] for n in names]
        ax[j].bar(x - bw / 2, inr, bw, label="trained region")
        ax[j].bar(x + bw / 2, out_, bw, label="held-out region")
        ax[j].set_xticks(x)
        ax[j].set_xticklabels([labels[n] for n in names])
        ax[j].set_title(title)
        ax[j].grid(axis="y", alpha=0.3)
        ax[j].legend()
    fig.suptitle("Spatial generalization: train on one region, forecast a never-seen region", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "region_holdout.png"), dpi=140)
    print("wrote region_holdout.png and region_holdout.json")
    for n in names:
        r = res[n]
        ret = f"{r['skill_retention']:.0%}" if r["skill_retention"] is not None else "n/a"
        print(f"{labels[n]:11s} skill: in-region {r['in_region']['skill']:+.3f}  "
              f"held-out {r['held_out_region']['skill']:+.3f}  (retention {ret})")


if __name__ == "__main__":
    main()
