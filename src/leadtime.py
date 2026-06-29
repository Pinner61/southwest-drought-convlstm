"""
Skill vs lead time.

v1.2 additions:
  - invalid-cell masking
  - lead-time curves for SPI-3, SPI-6, and SPI-12

    python src/leadtime.py --config config_demo.yaml --max_lead 6
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from models import build_model

DROUGHT_THR = -0.8
CHANNEL_NAMES = ["SPI-3", "SPI-6", "SPI-12"]


def rollout(model, window, K, device):
    """window: (B, L, C, H, W) -> stacked predictions (B, K, C, H, W)."""
    preds = []
    w = window.clone().to(device)
    with torch.no_grad():
        for _ in range(K):
            p = model(w)
            preds.append(p.unsqueeze(1))
            w = torch.cat([w[:, 1:], p.unsqueeze(1)], dim=1)
    return torch.cat(preds, dim=1).cpu().numpy()


def _masked(field, mask):
    if mask is None:
        return field.reshape(-1)
    return field[..., mask].reshape(-1)


def onset_f1(pred, actual, origin, mask=None):
    pf, af, of = _masked(pred, mask), _masked(actual, mask), _masked(origin, mask)
    pred_on = (of >= DROUGHT_THR) & (pf < DROUGHT_THR)
    true_on = (of >= DROUGHT_THR) & (af < DROUGHT_THR)
    tp = (pred_on & true_on).sum()
    fp = (pred_on & ~true_on).sum()
    fn = (~pred_on & true_on).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))


def skill_vs_persist(pred, actual, origin, mask=None):
    pf, af, of = _masked(pred, mask), _masked(actual, mask), _masked(origin, mask)
    persist_mse = ((of - af) ** 2).mean() + 1e-12
    return float(1 - ((pf - af) ** 2).mean() / persist_mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_lead", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = cfg["out_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = args.max_lead

    arr = np.load(os.path.join(out, "array.npy"))
    nz = np.load(os.path.join(out, "norm.npz"), allow_pickle=True)
    L, te, tr = int(nz["L"]), nz["te"], nz["tr"]
    valid_mask = nz["valid_mask"] if "valid_mask" in nz.files else (np.std(arr[:, 0], axis=0) > 1e-6)
    in_ch = arr.shape[1]
    clim = arr[tr.min():tr.max() + 1].mean(axis=0)

    origins = [t for t in te if t + K <= arr.shape[0]]
    window = np.stack([arr[t - L:t] for t in origins])
    actual = np.stack([arr[t:t + K] for t in origins])

    models = {}
    for name in ["pixel_lstm", "convlstm"]:
        m = build_model(name, in_ch, cfg["model"]).to(device)
        m.load_state_dict(torch.load(os.path.join(out, f"{name}.pt"), map_location=device))
        m.eval()
        models[name] = rollout(m, torch.from_numpy(window).float(), K, device)

    leads = np.arange(1, K + 1)
    all_curves = {}
    for ch in range(in_ch):
        cname = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"channel_{ch}"
        origin = window[:, -1, ch]
        curves = {"convlstm": {"skill": [], "onset": []},
                  "pixel_lstm": {"skill": [], "onset": []},
                  "climatology": {"skill": [], "onset": []}}
        for k in range(K):
            a = actual[:, k, ch]
            clim_pred = np.broadcast_to(clim[ch], a.shape)
            curves["climatology"]["skill"].append(skill_vs_persist(clim_pred, a, origin, valid_mask))
            curves["climatology"]["onset"].append(onset_f1(clim_pred, a, origin, valid_mask))
            for name in ["pixel_lstm", "convlstm"]:
                p = models[name][:, k, ch]
                curves[name]["skill"].append(skill_vs_persist(p, a, origin, valid_mask))
                curves[name]["onset"].append(onset_f1(p, a, origin, valid_mask))
        all_curves[cname] = curves

    json.dump({"leads": leads.tolist(), "curves_by_channel": all_curves},
              open(os.path.join(out, "leadtime_by_channel.json"), "w"), indent=2)
    # Backward compatible: SPI-3 only
    first_key = CHANNEL_NAMES[0] if in_ch >= 1 else "channel_0"
    json.dump({"leads": leads.tolist(), "curves": all_curves[first_key]},
              open(os.path.join(out, "leadtime.json"), "w"), indent=2)

    colors = {"convlstm": "#1b6ca8", "pixel_lstm": "#d98324", "climatology": "#888888"}
    labels = {"convlstm": "ConvLSTM", "pixel_lstm": "Pixel LSTM", "climatology": "Climatology"}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    curves = all_curves[first_key]
    for name in ["convlstm", "pixel_lstm", "climatology"]:
        ax[0].plot(leads, curves[name]["skill"], "-o", color=colors[name], label=labels[name])
        ax[1].plot(leads, curves[name]["onset"], "-o", color=colors[name], label=labels[name])
    ax[0].axhline(0, color="k", lw=1, ls="--")
    ax[0].text(leads[-1], 0.005, "persistence", ha="right", va="bottom", fontsize=8)
    ax[0].set_title(f"Forecast skill vs persistence ({first_key})")
    ax[0].set_ylabel("skill score (>0 beats persistence)")
    ax[1].set_title(f"Drought-onset detection ({first_key})")
    ax[1].set_ylabel("onset F1")
    for a in ax:
        a.set_xlabel("lead time (months)")
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle("How forecast skill changes with lead time", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "leadtime.png"), dpi=140)
    print("wrote leadtime.png, leadtime.json, and leadtime_by_channel.json")


if __name__ == "__main__":
    main()
