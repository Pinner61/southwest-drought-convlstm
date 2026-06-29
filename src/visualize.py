"""
Make project figures.

v1.2 masks invalid grid cells before plotting so out-of-domain edge artifacts do
not appear as real drought values.

    python src/visualize.py --config config_demo.yaml
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def mask_invalid(x, mask):
    y = np.array(x, dtype=float, copy=True)
    if mask is not None:
        y[..., ~mask] = np.nan
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = cfg["out_dir"]

    s = np.load(os.path.join(out, "samples.npz"), allow_pickle=True)
    actual, conv, persist = s["actual"], s["convlstm"], s["persistence"]
    mask = s["valid_mask"] if "valid_mask" in s.files else np.ones(actual.shape[-2:], dtype=bool)

    actual_m = mask_invalid(actual, mask)
    conv_m = mask_invalid(conv, mask)
    persist_m = mask_invalid(persist, mask)

    vmin, vmax = -2.5, 2.5
    cmap = plt.get_cmap("BrBG").copy()
    cmap.set_bad(color="lightgray")

    n = min(5, len(actual_m))
    idx = np.linspace(0, len(actual_m) - 1, n).astype(int)
    fig, ax = plt.subplots(3, n, figsize=(2.4 * n, 7.2))
    rows = [("Actual", actual_m), ("ConvLSTM", conv_m), ("Persistence", persist_m)]
    for r, (label, data) in enumerate(rows):
        for c, t in enumerate(idx):
            ax[r, c].imshow(data[t], cmap=cmap, vmin=vmin, vmax=vmax)
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f"test month {t}", fontsize=9)
            if c == 0:
                ax[r, c].set_ylabel(label, fontsize=11)
    fig.suptitle("Drought field: actual vs forecast (gray = masked invalid cells)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "panels.png"), dpi=130)
    print("wrote panels.png")

    fig2, ax2 = plt.subplots(1, 2, figsize=(7, 3.6))
    im0 = ax2[0].imshow(actual_m[0], cmap=cmap, vmin=vmin, vmax=vmax)
    im1 = ax2[1].imshow(conv_m[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax2[0].set_title("Actual")
    ax2[1].set_title("ConvLSTM forecast")
    for a in ax2:
        a.set_xticks([])
        a.set_yticks([])

    def upd(t):
        im0.set_data(actual_m[t])
        im1.set_data(conv_m[t])
        fig2.suptitle(f"test month {t}")
        return im0, im1

    ani = animation.FuncAnimation(fig2, upd, frames=len(actual_m), interval=200, blit=False)
    ani.save(os.path.join(out, "spread.gif"), writer="pillow", dpi=90)
    print("wrote spread.gif")


if __name__ == "__main__":
    main()
