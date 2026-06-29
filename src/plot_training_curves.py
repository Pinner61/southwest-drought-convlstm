"""
Plot training and validation losses saved by src/train.py.

    python src/plot_training_curves.py --config config_demo.yaml
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import yaml


def load_history(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = cfg["out_dir"]

    names = [("convlstm", "ConvLSTM"), ("pixel_lstm", "Pixel LSTM")]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    found = False
    for name, label in names:
        path = os.path.join(out, f"{name}_history.json")
        if not os.path.exists(path):
            print(f"missing {path}; run src/train.py first")
            continue
        hist = load_history(path)
        epochs = [h["epoch"] for h in hist]
        train = [h["train"] for h in hist]
        val = [h["val"] for h in hist]
        ax.plot(epochs, train, "--", label=f"{label} train")
        ax.plot(epochs, val, "-", label=f"{label} validation")
        found = True

    if not found:
        raise SystemExit("No history files found. Run training first.")

    ax.set_title("Training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out, "training_curves.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
