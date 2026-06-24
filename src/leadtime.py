"""
Skill vs lead time.

The one-step result (next month) is a single point. The research question is how
forecast skill behaves as the horizon lengthens: does the ConvLSTM's spatial
advantage over persistence and the per-pixel LSTM grow, shrink, or hold as we
push the forecast further out?

We roll each learned model forward AUTOREGRESSIVELY (predict month t+1, feed the
prediction back in, predict t+2, ...). That tests whether the model has learned
genuine field dynamics, not just a one-step correlation. Persistence and
climatology hold their last value across all leads.

    python src/leadtime.py --config config_demo.yaml
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


def rollout(model, window, K, device):
    """window: (B, L, C, H, W) -> stacked predictions (B, K, C, H, W)."""
    preds = []
    w = window.clone().to(device)
    with torch.no_grad():
        for _ in range(K):
            p = model(w)                       # (B, C, H, W)
            preds.append(p.unsqueeze(1))
            w = torch.cat([w[:, 1:], p.unsqueeze(1)], dim=1)
    return torch.cat(preds, dim=1).cpu().numpy()


def onset_f1(pred0, actual0, origin0):
    pred_on = (origin0 >= DROUGHT_THR) & (pred0 < DROUGHT_THR)
    true_on = (origin0 >= DROUGHT_THR) & (actual0 < DROUGHT_THR)
    tp = (pred_on & true_on).sum()
    fp = (pred_on & ~true_on).sum()
    fn = (~pred_on & true_on).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))


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
    in_ch = arr.shape[1]
    clim = arr[tr.min():tr.max() + 1].mean(axis=0)

    origins = [t for t in te if t + K <= arr.shape[0]]
    window = np.stack([arr[t - L:t] for t in origins])            # (B, L, C, H, W)
    actual = np.stack([arr[t:t + K] for t in origins])            # (B, K, C, H, W)
    origin0 = window[:, -1, 0]                                    # last observed field

    models = {}
    for name in ["pixel_lstm", "convlstm"]:
        m = build_model(name, in_ch, cfg["model"]).to(device)
        m.load_state_dict(torch.load(os.path.join(out, f"{name}.pt"), map_location=device))
        m.eval()
        models[name] = rollout(m, torch.from_numpy(window).float(), K, device)

    leads = np.arange(1, K + 1)
    curves = {"convlstm": {"skill": [], "onset": []},
              "pixel_lstm": {"skill": [], "onset": []},
              "climatology": {"skill": [], "onset": []}}

    for k in range(K):
        a0 = actual[:, k, 0]
        persist_mse = ((origin0 - a0) ** 2).mean()
        clim_pred0 = np.broadcast_to(clim[0], a0.shape)
        curves["climatology"]["skill"].append(float(1 - ((clim_pred0 - a0) ** 2).mean() / persist_mse))
        curves["climatology"]["onset"].append(onset_f1(clim_pred0, a0, origin0))
        for name in ["pixel_lstm", "convlstm"]:
            p0 = models[name][:, k, 0]
            curves[name]["skill"].append(float(1 - ((p0 - a0) ** 2).mean() / persist_mse))
            curves[name]["onset"].append(onset_f1(p0, a0, origin0))

    json.dump({"leads": leads.tolist(), "curves": curves},
              open(os.path.join(out, "leadtime.json"), "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"convlstm": "#1b6ca8", "pixel_lstm": "#d98324", "climatology": "#888888"}
    labels = {"convlstm": "ConvLSTM", "pixel_lstm": "Pixel LSTM", "climatology": "Climatology"}
    for name in ["convlstm", "pixel_lstm", "climatology"]:
        ax[0].plot(leads, curves[name]["skill"], "-o", color=colors[name], label=labels[name])
        ax[1].plot(leads, curves[name]["onset"], "-o", color=colors[name], label=labels[name])
    ax[0].axhline(0, color="k", lw=1, ls="--")
    ax[0].text(leads[-1], 0.005, "persistence", ha="right", va="bottom", fontsize=8)
    ax[0].set_title("Forecast skill vs persistence"); ax[0].set_ylabel("skill score (>0 beats persistence)")
    ax[1].set_title("Drought-onset detection (F1)"); ax[1].set_ylabel("onset F1")
    for a in ax:
        a.set_xlabel("lead time (months)"); a.legend(); a.grid(alpha=0.3)
    fig.suptitle("How forecast skill degrades with lead time", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "leadtime.png"), dpi=140)
    print("wrote leadtime.png and leadtime.json")
    for name in ["convlstm", "pixel_lstm", "climatology"]:
        print(f"{name:12s} skill@1={curves[name]['skill'][0]:+.3f}  skill@{K}={curves[name]['skill'][-1]:+.3f}")


if __name__ == "__main__":
    main()
