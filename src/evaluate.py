"""
Evaluate every model on the held-out TEST period.

v1.2 additions:
  - invalid-cell masking for edge cells outside valid gridMET coverage
  - separate SPI-3, SPI-6, and SPI-12 evaluation
  - metrics_by_channel.json for the dashboard and report updates

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

DROUGHT_THR = -0.8
CHANNEL_NAMES = ["SPI-3", "SPI-6", "SPI-12"]


def collect_targets(arr, L, te):
    xs, ys, prev = [], [], []
    for t in te:
        xs.append(arr[t - L:t])
        ys.append(arr[t])
        prev.append(arr[t - 1])
    return np.array(xs), np.array(ys), np.array(prev)


def _apply_mask(field, mask):
    if mask is None:
        return field.reshape(-1)
    return field[..., mask].reshape(-1)


def metrics(pred, y, prev, channel, mask=None, threshold=DROUGHT_THR):
    p0, y0, pr0 = pred[:, channel], y[:, channel], prev[:, channel]
    pf = _apply_mask(p0, mask)
    yf = _apply_mask(y0, mask)
    prf = _apply_mask(pr0, mask)

    mse_model = ((pf - yf) ** 2).mean()
    mse_persist = ((prf - yf) ** 2).mean() + 1e-12
    rmse = np.sqrt(mse_model)
    mae = np.abs(pf - yf).mean()
    skill = 1 - mse_model / mse_persist

    pc = pf - pf.mean()
    yc = yf - yf.mean()
    acc = float((pc * yc).sum() / (np.sqrt((pc ** 2).sum() * (yc ** 2).sum()) + 1e-9))

    pred_onset = (prf >= threshold) & (pf < threshold)
    true_onset = (prf >= threshold) & (yf < threshold)
    tp = (pred_onset & true_onset).sum()
    fp = (pred_onset & ~true_onset).sum()
    fn = (~pred_onset & true_onset).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    return {k: float(v) for k, v in dict(
        RMSE=rmse, MAE=mae, skill_vs_persist=skill,
        anomaly_corr=acc, onset_F1=f1, onset_recall=rec,
        onset_count=true_onset.sum()).items()}


def print_table(results, channel_name):
    cols = ["RMSE", "MAE", "skill_vs_persist", "anomaly_corr", "onset_F1", "onset_recall"]
    order = ["persistence", "climatology", "pixel_lstm", "convlstm"]
    w = 16
    print(f"\n=== {channel_name} ===")
    print("model".ljust(w) + "".join(c.rjust(w) for c in cols))
    print("-" * (w * (len(cols) + 1)))
    for m in order:
        row = results[m]
        print(m.ljust(w) + "".join(f"{row[c]:>{w}.3f}" for c in cols))


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
    valid_mask = nz["valid_mask"] if "valid_mask" in nz.files else (np.std(arr[:, 0], axis=0) > 1e-6)
    in_ch = arr.shape[1]

    X, Y, PREV = collect_targets(arr, L, te)
    clim = arr[tr.min():tr.max() + 1].mean(axis=0)
    clim_pred = np.repeat(clim[None], len(Y), axis=0)

    predictions = {"persistence": PREV, "climatology": clim_pred}

    loader = DataLoader(SeqDataset(arr, L, te), batch_size=cfg["train"]["batch"])
    for name in ["pixel_lstm", "convlstm"]:
        model = build_model(name, in_ch, cfg["model"]).to(device)
        model.load_state_dict(torch.load(os.path.join(out, f"{name}.pt"), map_location=device))
        model.eval()
        outs = []
        with torch.no_grad():
            for x, _ in loader:
                outs.append(model(x.to(device)).cpu().numpy())
        predictions[name] = np.concatenate(outs)

    channel_results = {}
    for ch in range(in_ch):
        channel_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"channel_{ch}"
        res = {name: metrics(P, Y, PREV, ch, valid_mask) for name, P in predictions.items()}
        channel_results[channel_name] = res
        print_table(res, channel_name)

    # Keep metrics.json backward compatible with the original report: SPI-3 only.
    spi3_key = CHANNEL_NAMES[0] if in_ch >= 1 else "channel_0"
    json.dump(channel_results[spi3_key], open(os.path.join(out, "metrics.json"), "w"), indent=2)
    json.dump(channel_results, open(os.path.join(out, "metrics_by_channel.json"), "w"), indent=2)

    np.savez_compressed(os.path.join(out, "samples.npz"),
                        actual=Y[:, 0], persistence=PREV[:, 0],
                        convlstm=predictions["convlstm"][:, 0],
                        pixel_lstm=predictions["pixel_lstm"][:, 0],
                        actual_all=Y,
                        persistence_all=PREV,
                        convlstm_all=predictions["convlstm"],
                        pixel_lstm_all=predictions["pixel_lstm"],
                        valid_mask=valid_mask)
    print(f"\nwrote metrics.json, metrics_by_channel.json, and samples.npz to {out}")


if __name__ == "__main__":
    main()
