"""
Synthetic drought-index fields used to verify the pipeline end-to-end without
network access to real climate servers.

The generator is deliberately *not* trivial: it produces fields with both
temporal autocorrelation (AR(1)) and a coherent negative anomaly ("drought
blob") that drifts across the grid over time. That drift is genuine spatial
propagation, so a model that sees the whole field (ConvLSTM) should beat a
model that treats each pixel independently (pixel-wise LSTM). If the evaluation
harness can detect that gap on synthetic data, it will detect it on real data.
"""
import numpy as np


def _blur(field, passes=2):
    """Cheap separable ~Gaussian smoothing (avoids a scipy dependency)."""
    k = np.array([0.25, 0.5, 0.25])
    out = field.astype(float)
    for _ in range(passes):
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
    return out


def make_synthetic(T=180, H=24, W=24, C=2, seed=0):
    """
    Returns an array of shape (T, C, H, W) of standardized drought-like indices.
    Channel 0 ~ short-timescale index (e.g. SPI-3), channel 1 ~ a smoothed,
    lagged longer-timescale index (e.g. SPI-12).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)

    base = np.zeros((T, H, W))
    prev = _blur(rng.standard_normal((H, W)), passes=3)
    for t in range(T):
        innovation = _blur(rng.standard_normal((H, W)), passes=3)
        prev = 0.7 * prev + 0.7 * innovation          # AR(1) in time, smooth in space
        base[t] = prev

    # A negative anomaly that travels diagonally across the grid: the "drought".
    for t in range(T):
        cx = (W - 4) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 60.0))
        cy = (H - 4) * (t % 90) / 90.0
        amp = 2.2 * (0.6 + 0.4 * np.sin(2 * np.pi * t / 45.0))
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0 ** 2)))
        base[t] -= amp * blob

    ch0 = base
    ch1 = np.zeros_like(base)                          # slower / lagged channel
    acc = base[0].copy()
    for t in range(T):
        acc = 0.85 * acc + 0.15 * base[t]
        ch1[t] = _blur(acc, passes=1)

    arr = np.stack([ch0, ch1], axis=1) if C == 2 else ch0[:, None]
    # standardize per channel (real pipeline standardizes per pixel on train stats)
    for c in range(arr.shape[1]):
        arr[:, c] = (arr[:, c] - arr[:, c].mean()) / (arr[:, c].std() + 1e-8)
    return arr.astype(np.float32)
