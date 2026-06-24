"""
Region-holdout: does the model generalize to space it never trained on?

We split the domain into two spatially disjoint regions. Both models are trained
ONLY on region A, then evaluated on:
  - region A during the held-out test period   (in-region generalization)
  - region B during the held-out test period   (spatial transfer: never-seen space)

A ConvLSTM's convolutional gates are translation-equivariant, so if it has
learned genuine local drought dynamics rather than memorizing pixel locations,
its skill on region B should be close to its skill on region A. That is the
result that answers "does it generalize?".

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


def onset_f1(pred0, actual0, prev0):
    pred_on = (prev0 >= DROUGHT_THR) & (pred0 < DROUGHT_THR)
    true_on = (prev0 >= DROUGHT_THR) & (actual0 < DROUGHT_THR)
    tp = (pred_on & true_on).sum(); fp = (pred_on & ~true_on).sum(); fn = (~pred_on & true_on).sum()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))


def train_on_region(arr, L, tr, cfg, device):
    models = {}
    for name in ["convlstm", "pixel_lstm"]:
        m = build_model(name, arr.shape[1], cfg["model"]).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=cfg["train"]["lr"])
        loader = DataLoader(SeqDataset(arr, L, tr), batch_size=cfg["train"]["batch"], shuffle=True)
        for _ in range(cfg["train"]["epochs"]):
            m.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = ((m(x) - y) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        models[name] = m
    return models


def eval_region(model, arr, L, times, device):
    X = np.stack([arr[t - L:t] for t in times])
    Y = np.stack([arr[t] for t in times])
    PREV = np.stack([arr[t - 1] for t in times])
    with torch.no_grad():
        P = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    mse_persist = ((PREV[:, 0] - Y[:, 0]) ** 2).mean()
    skill = float(1 - ((P[:, 0] - Y[:, 0]) ** 2).mean() / mse_persist)
    return skill, onset_f1(P[:, 0], Y[:, 0], PREV[:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = cfg["out_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(cfg.get("seed", 0)); torch.manual_seed(cfg.get("seed", 0))

    arr, _ = build_array(cfg)
    L = cfg["model"]["seq_len"]
    tr, va, te = chronological_splits(arr.shape[0], L,
                                      cfg["data"].get("frac_train", 0.7),
                                      cfg["data"].get("frac_val", 0.15))
    arr_std, _, _ = standardize(arr, tr)

    W = arr_std.shape[-1]
    cut = W // 2
    region_A = arr_std[..., :cut].copy()      # training region
    region_B = arr_std[..., cut:].copy()      # held-out region (never trained on)

    models = train_on_region(region_A, L, tr, cfg, device)

    res = {}
    for name, m in models.items():
        sA, fA = eval_region(m, region_A, L, te, device)
        sB, fB = eval_region(m, region_B, L, te, device)
        res[name] = {"in_region": {"skill": sA, "onset_F1": fA},
                     "held_out_region": {"skill": sB, "onset_F1": fB},
                     "skill_retention": float(sB / sA) if sA > 0 else None}
    json.dump(res, open(os.path.join(out, "region_holdout.json"), "w"), indent=2)

    names = ["convlstm", "pixel_lstm"]
    labels = {"convlstm": "ConvLSTM", "pixel_lstm": "Pixel LSTM"}
    x = np.arange(len(names)); bw = 0.36
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for j, metric, title in [(0, "skill", "Skill vs persistence"), (1, "onset_F1", "Drought-onset F1")]:
        inr = [res[n]["in_region"][metric] for n in names]
        out_ = [res[n]["held_out_region"][metric] for n in names]
        ax[j].bar(x - bw / 2, inr, bw, label="trained region", color="#1b6ca8")
        ax[j].bar(x + bw / 2, out_, bw, label="held-out region", color="#a8472f")
        ax[j].set_xticks(x); ax[j].set_xticklabels([labels[n] for n in names])
        ax[j].set_title(title); ax[j].grid(axis="y", alpha=0.3); ax[j].legend()
    fig.suptitle("Spatial generalization: train on one region, forecast a never-seen region", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "region_holdout.png"), dpi=140)
    print("wrote region_holdout.png and region_holdout.json")
    for n in names:
        r = res[n]
        ret = f"{r['skill_retention']:.0%}" if r["skill_retention"] else "n/a"
        print(f"{labels[n]:11s} skill: in-region {r['in_region']['skill']:+.3f}  "
              f"held-out {r['held_out_region']['skill']:+.3f}  (retention {ret})")


if __name__ == "__main__":
    main()
