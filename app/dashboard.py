"""
Streamlit dashboard for exploring drought forecast outputs.

Run from the repository root:
    streamlit run app/dashboard.py -- --out outputs_real
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

CHANNELS = ["SPI-3", "SPI-6", "SPI-12"]
MODELS = {
    "ConvLSTM": "convlstm_all",
    "Pixel LSTM": "pixel_lstm_all",
    "Persistence": "persistence_all",
}


def load_json(path):
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_field(field, mask, title):
    arr = np.array(field, dtype=float, copy=True)
    if mask is not None:
        arr[~mask] = np.nan
    cmap = plt.get_cmap("BrBG").copy()
    cmap.set_bad("lightgray")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = ax.imshow(arr, cmap=cmap, vmin=-2.5, vmax=2.5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Standardized drought index")
    st.pyplot(fig, clear_figure=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", default="outputs_real")
    args, _ = parser.parse_known_args()
    out = Path(args.out)

    st.set_page_config(page_title="Drought ConvLSTM Dashboard", layout="wide")
    st.title("Drought Onset Forecasting Dashboard")
    st.caption("Explore held-out forecast fields, metrics by SPI channel, lead-time curves, and spatial holdout results.")

    samples_path = out / "samples.npz"
    if not samples_path.exists():
        st.error(f"Could not find {samples_path}. Run train.py and evaluate.py first.")
        st.code("python src/train.py --config config.yaml\npython src/evaluate.py --config config.yaml")
        return

    s = np.load(samples_path, allow_pickle=True)
    actual_all = s["actual_all"] if "actual_all" in s.files else s["actual"][:, None]
    mask = s["valid_mask"] if "valid_mask" in s.files else np.ones(actual_all.shape[-2:], dtype=bool)

    metrics = load_json(out / "metrics_by_channel.json")
    lead = load_json(out / "leadtime_by_channel.json")
    region = load_json(out / "region_holdout.json")

    st.sidebar.header("Controls")
    t = st.sidebar.slider("Test month index", 0, actual_all.shape[0] - 1, min(5, actual_all.shape[0] - 1))
    ch = st.sidebar.selectbox("Channel", list(range(actual_all.shape[1])), format_func=lambda i: CHANNELS[i] if i < len(CHANNELS) else f"channel {i}")
    model_label = st.sidebar.selectbox("Forecast", list(MODELS.keys()))

    st.subheader("Forecast field comparison")
    c1, c2 = st.columns(2)
    with c1:
        render_field(actual_all[t, ch], mask, f"Actual {CHANNELS[ch] if ch < len(CHANNELS) else ch} | test month {t}")
    with c2:
        key = MODELS[model_label]
        if key in s.files:
            render_field(s[key][t, ch], mask, f"{model_label} forecast | test month {t}")
        else:
            st.warning(f"{key} not found in samples.npz. Re-run evaluate.py with the v1.2 code.")

    st.subheader("Model metrics by channel")
    if metrics:
        selected = CHANNELS[ch] if ch < len(CHANNELS) else f"channel_{ch}"
        if selected in metrics:
            st.dataframe(metrics[selected], use_container_width=True)
        else:
            st.json(metrics)
    else:
        st.info("metrics_by_channel.json not found. Run src/evaluate.py.")

    st.subheader("Lead-time curves")
    if lead and "curves_by_channel" in lead:
        selected = CHANNELS[ch] if ch < len(CHANNELS) else f"channel_{ch}"
        curves = lead["curves_by_channel"].get(selected)
        if curves:
            leads = lead["leads"]
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
            for name, label in [("convlstm", "ConvLSTM"), ("pixel_lstm", "Pixel LSTM"), ("climatology", "Climatology")]:
                ax[0].plot(leads, curves[name]["skill"], marker="o", label=label)
                ax[1].plot(leads, curves[name]["onset"], marker="o", label=label)
            ax[0].set_title("Skill vs persistence")
            ax[1].set_title("Onset F1")
            for a in ax:
                a.set_xlabel("Lead time (months)")
                a.grid(alpha=0.3)
                a.legend()
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
    else:
        st.info("leadtime_by_channel.json not found. Run src/leadtime.py.")

    st.subheader("Region holdout")
    if region:
        st.json(region)
        img = out / "region_holdout.png"
        if img.exists():
            st.image(str(img))
    else:
        st.info("region_holdout.json not found. Run src/region_holdout.py.")


if __name__ == "__main__":
    main()
