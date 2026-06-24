"""
Evaluate every model on the held-out TEST period against the baselines that
make this a real study:

  persistence   - next month = this month (the bar every forecast must clear)
  climatology   - per-pixel mean of the training period
  pixel_lstm    - temporal model, no spatial coupling
  convlstm      - spatiotemporal model (the hypothesis)

Metrics: RMSE, MAE, skill score vs persistence (1 - MSE/MSE_persist; >0 means
better than persistence), anomaly correlation, and drought-ONSET F1 (did the
model predict a pixel crossing into drought next month). The onset metric is
what hydrologists actually care about and is where a propagation-aware model
should earn its keep.

    python src/evaluate.py --config config_demo.yaml
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import SeqDataset
from models import build_model

DROUGHT_THR = -0.8   # ~ SPI/SPEI "moderate drought" (D1) boundary


def collect_targets(arr, L, te):
    xs, ys, prev = [], [], []
    for t in te:
        xs.append(arr[t - L:t])
        ys.append(arr[t])
        prev.append(arr[t - 1])
    return np.array(xs), np.array(ys), np.array(prev)


def metrics(pred, y, prev, mse_persist):
    p0, y0, pr0 = pred[:, 0], y[:, 0], prev[:, 0]   # evaluate on channel 0
    rmse = np.sqrt(((p0 - y0) ** 2).mean())
    mae = np.abs(p0 - y0).mean()
    skill = 1 - ((p0 - y0) ** 2).mean() / mse_persist
    pf, yf = p0.ravel() - p0.mean(), y0.ravel() - y0.mean()
    acc = float((pf * yf).sum() / (np.sqrt((pf ** 2).sum() * (yf ** 2).sum()) + 1e-9))

    pred_onset = (pr0 >= DROUGHT_THR) & (p0 < DROUGHT_THR)
    true_onset = (pr0 >= DROUGHT_THR) & (y0 < DROUGHT_THR)
    tp = (pred_onset & true_onset).sum()
    fp = (pred_onset & ~true_onset).sum()
    fn = (~pred_onset & true_onset).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {k: float(v) for k, v in dict(
        RMSE=rmse, MAE=mae, skill_vs_persist=skill,
        anomaly_corr=acc, onset_F1=f1, onset_recall=rec).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = cfg["out_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    arr = np.load(os.path.join(out, "array.npy"))
    nz = np.load(os.path.join(out, "norm.npz"), allow_pickle=True)
    L, te, tr = int(nz["L"]), nz["te"], nz["tr"]
    in_ch = arr.shape[1]

    X, Y, PREV = collect_targets(arr, L, te)
    mse_persist = ((PREV[:, 0] - Y[:, 0]) ** 2).mean()

    results, preds_for_viz = {}, {}

    # --- baselines ---
    results["persistence"] = metrics(PREV, Y, PREV, mse_persist)
    clim = arr[tr.min():tr.max() + 1].mean(axis=0)          # per-pixel train mean
    clim_pred = np.repeat(clim[None], len(Y), axis=0)
    results["climatology"] = metrics(clim_pred, Y, PREV, mse_persist)

    # --- learned models ---
    loader = DataLoader(SeqDataset(arr, L, te), batch_size=cfg["train"]["batch"])
    for name in ["pixel_lstm", "convlstm"]:
        model = build_model(name, in_ch, cfg["model"]).to(device)
        model.load_state_dict(torch.load(os.path.join(out, f"{name}.pt"), map_location=device))
        model.eval()
        outs = []
        with torch.no_grad():
            for x, _ in loader:
                outs.append(model(x.to(device)).cpu().numpy())
        P = np.concatenate(outs)
        results[name] = metrics(P, Y, PREV, mse_persist)
        preds_for_viz[name] = P[:, 0]

    json.dump(results, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    np.savez(os.path.join(out, "samples.npz"),
             actual=Y[:, 0], persistence=PREV[:, 0], **preds_for_viz)

    # --- table ---
    cols = ["RMSE", "MAE", "skill_vs_persist", "anomaly_corr", "onset_F1", "onset_recall"]
    order = ["persistence", "climatology", "pixel_lstm", "convlstm"]
    w = 16
    print("\n" + "model".ljust(w) + "".join(c.rjust(w) for c in cols))
    print("-" * (w * (len(cols) + 1)))
    for m in order:
        row = results[m]
        print(m.ljust(w) + "".join(f"{row[c]:>{w}.3f}" for c in cols))


if __name__ == "__main__":
    main()
