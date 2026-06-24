"""
Data handling.

build_array(cfg) returns a single numpy array of shape (T, C, H, W) plus a
list of timestamps. Two sources:

  source: synthetic   -> generated locally, used to verify the pipeline.
  source: gridmet     -> real data. Downloads gridMET precipitation and
                         reference ET over a bounding box via OPeNDAP, computes
                         SPI/SPEI with the `climate_indices` package, and stacks
                         the indices as channels.

The real path is the ONLY step that needs network access to climate servers,
so it lives behind a clear boundary and is run on your laptop or SOL, not in
the build sandbox.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Windowing + chronological split
# --------------------------------------------------------------------------- #
class SeqDataset(Dataset):
    """Sliding windows of L past frames -> the next frame."""

    def __init__(self, arr, L, target_times):
        self.arr = arr
        self.L = L
        self.t = list(target_times)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        t = self.t[i]
        x = self.arr[t - self.L:t]      # (L, C, H, W)
        y = self.arr[t]                 # (C, H, W)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def chronological_splits(T, L, frac_train=0.7, frac_val=0.15):
    """Strictly time-ordered target indices. No shuffling, no leakage."""
    valid = np.arange(L, T)
    n = len(valid)
    a = int(n * frac_train)
    b = int(n * (frac_train + frac_val))
    return valid[:a], valid[a:b], valid[b:]


def standardize(arr, train_times):
    """Per-pixel, per-channel standardization using TRAIN-period stats only."""
    train = arr[train_times.min():train_times.max() + 1]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return ((arr - mean) / std).astype(np.float32), mean, std


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def build_array(cfg):
    src = cfg["data"]["source"]
    if src == "synthetic":
        from synth import make_synthetic
        d = cfg["data"]
        arr = make_synthetic(d.get("T", 180), d.get("H", 24), d.get("W", 24),
                             d.get("channels", 2), cfg.get("seed", 0))
        times = list(range(arr.shape[0]))
        return arr, times
    if src == "gridmet":
        return _build_gridmet(cfg)
    raise ValueError(src)


def _build_gridmet(cfg):
    """
    Real data path. Requires: xarray, netCDF4, dask, climate-indices.

    Downloads gridMET monthly precipitation over the configured bounding box,
    coarsens it spatially (to keep compute tractable on a single node), and
    derives SPI at 3-, 6-, and 12-month scales as the three input channels.
    SPI is the WMO-recommended meteorological drought index and only needs
    precipitation, which keeps the pipeline robust.
    """
    import xarray as xr
    from climate_indices import indices, compute

    d = cfg["data"]
    lat0, lat1 = d["bbox"]["lat"]     # e.g. [31.0, 37.0]  (Southwest US)
    lon0, lon1 = d["bbox"]["lon"]     # e.g. [-115.0, -107.0]
    y0, y1 = d["years"]               # e.g. [1990, 2024]
    coarsen = int(d.get("coarsen", 4))
    base = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/"

    pr = xr.open_dataset(base + "agg_met_pr_1979_CurrentYear_CONUS.nc#fillmismatch")
    precip = pr["precipitation_amount"].sel(
        lat=slice(lat1, lat0), lon=slice(lon0, lon1),
        day=slice(f"{y0}-01-01", f"{y1}-12-31"))
    precip = precip.resample(day="1MS").sum().rename(day="time")   # monthly totals
    if coarsen and coarsen > 1:
        precip = precip.coarsen(lat=coarsen, lon=coarsen, boundary="trim").mean()
    precip = precip.compute()
    print(f"gridMET precip loaded: {dict(precip.sizes)}")

    T = precip.sizes["time"]; H = precip.sizes["lat"]; W = precip.sizes["lon"]
    vals = precip.values.reshape(T, H * W)

    def spi_grid(scale):
        out = np.full((T, H * W), np.nan)
        for p in range(H * W):
            col = vals[:, p]
            if np.isnan(col).all():
                continue
            try:
                out[:, p] = indices.spi(
                    np.nan_to_num(col).astype(float), scale,
                    indices.Distribution.gamma, y0, y0, y1,
                    compute.Periodicity.monthly)
            except Exception:
                pass
        print(f"  SPI-{scale} computed")
        return out.reshape(T, H, W)

    arr = np.stack([spi_grid(3), spi_grid(6), spi_grid(12)], axis=1).astype(np.float32)
    arr = np.nan_to_num(arr)
    times = list(np.datetime_as_string(precip["time"].values, unit="M"))
    return arr, times
