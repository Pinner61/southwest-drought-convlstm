"""
Train the learned models (ConvLSTM and the pixel-wise LSTM baseline).

    python src/train.py --config config_demo.yaml     # synthetic, fast
    python src/train.py --config config.yaml          # real gridMET data

Persistence and climatology need no training; they are computed in evaluate.py.
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import build_array, chronological_splits, standardize, SeqDataset
from models import build_model


def run_epoch(model, loader, opt, device, train=True):
    model.train(train)
    total, n = 0.0, 0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed = cfg.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)

    arr, times = build_array(cfg)
    L = cfg["model"]["seq_len"]
    tr, va, te = chronological_splits(arr.shape[0], L,
                                      cfg["data"].get("frac_train", 0.7),
                                      cfg["data"].get("frac_val", 0.15))
    arr_std, mean, std = standardize(arr, tr)
    np.savez(os.path.join(out, "norm.npz"), mean=mean, std=std,
             tr=tr, va=va, te=te, L=L)
    np.save(os.path.join(out, "array.npy"), arr_std)

    in_ch = arr_std.shape[1]
    tl = DataLoader(SeqDataset(arr_std, L, tr), batch_size=cfg["train"]["batch"], shuffle=True)
    vl = DataLoader(SeqDataset(arr_std, L, va), batch_size=cfg["train"]["batch"])

    for name in ["convlstm", "pixel_lstm"]:
        model = build_model(name, in_ch, cfg["model"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
        best, best_state, hist = float("inf"), None, []
        for ep in range(cfg["train"]["epochs"]):
            trl = run_epoch(model, tl, opt, device, True)
            val = run_epoch(model, vl, opt, device, False)
            hist.append({"epoch": ep, "train": trl, "val": val})
            if val < best:
                best, best_state = val, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{name}] epoch {ep:02d}  train {trl:.4f}  val {val:.4f}")
        torch.save(best_state, os.path.join(out, f"{name}.pt"))
        json.dump(hist, open(os.path.join(out, f"{name}_history.json"), "w"), indent=2)
        print(f"[{name}] best val {best:.4f} -> saved\n")


if __name__ == "__main__":
    main()
