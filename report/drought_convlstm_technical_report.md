# Forecasting Drought Propagation in the U.S. Southwest with Spatiotemporal Deep Learning

**A reproducible ConvLSTM versus pixel-LSTM baseline study using SPI drought fields**  
**Tejas Sharma**  
Computer Science, Arizona State University  
Technical Report v1.1 · June 2026

---

## Abstract

Drought conditions are spatial as well as temporal: precipitation deficits persist, expand, and shift across neighboring regions rather than evolving independently at every grid cell. This project tests whether a model that observes the full drought-index field improves monthly drought forecasting compared with a model that treats each location independently. Using gridMET precipitation from 1990 to 2024, I derived SPI-3, SPI-6, and SPI-12 fields over a U.S. Southwest domain (31-37°N, 115-107°W) and trained two sequence models: a ConvLSTM that preserves spatial structure and a shared per-pixel LSTM that uses only each grid cell's own history. Both models predict the next monthly drought field from six months of prior SPI fields and are evaluated against persistence and climatology using a chronological train/validation/test split. On one-month-ahead SPI-3 field metrics, the per-pixel LSTM has slightly lower RMSE than the ConvLSTM (0.694 vs. 0.715), but the ConvLSTM improves drought-onset detection (F1 0.231 vs. 0.216; recall 0.159 vs. 0.141). In autoregressive 1-6 month rollouts, the ConvLSTM maintains higher onset F1 at every lead. These results suggest that spatial context is most useful for identifying newly emerging drought, not necessarily for minimizing average field reconstruction error.

**Keywords:** drought forecasting; ConvLSTM; SPI; gridMET; spatiotemporal machine learning; climate data; reproducible research

## 1. Research Question and Contribution

Drought is often measured with single-location indices such as the Standardized Precipitation Index (SPI), but the process being measured is not isolated to a single pixel. Dry anomalies can persist locally, expand from one part of a region to another, or appear as part of broader circulation-driven precipitation deficits. A model that only sees each grid cell's own time series cannot use neighboring information, even when nearby cells contain signals about the next month.

This project asks a narrow question:

> Does spatial context help a sequence model forecast monthly drought fields, and does the answer depend on whether performance is measured by bulk field accuracy or drought-onset detection?

The contribution is a reproducible baseline study rather than an operational drought model. The repository implements a full pipeline for downloading gridMET precipitation, deriving SPI fields, training a ConvLSTM and a per-pixel LSTM under the same split, comparing them with persistence and climatology, and visualizing both one-step and multi-lead forecasts.

The most important result is not that one architecture is uniformly better. The per-pixel LSTM is slightly stronger on RMSE and MAE, while the ConvLSTM is stronger on onset F1 and recall. This distinction is useful because drought-warning applications care about where drought is newly emerging, not only average error across all pixels.

## 2. Data and Study Design

Precipitation data come from gridMET (Abatzoglou, 2013), a daily gridded meteorological dataset over the contiguous United States. The pipeline aggregates daily precipitation to monthly totals and uses the `climate_indices` package to compute SPI at 3-, 6-, and 12-month accumulation scales. These three SPI fields are stacked as model channels.

The study domain is a U.S. Southwest bounding box spanning approximately 31-37°N and 115-107°W. The record covers January 1990 through December 2024, giving 420 monthly observations. The native gridMET resolution is spatially coarsened by a factor of 4 before SPI computation to keep the experiment tractable on a single CPU/GPU node. Based on the nominal gridMET resolution and configured domain, this corresponds to roughly 16 km cells, although the repository does not store a final logged array shape for the completed run.

**Table 1. Experimental setup.**

| Component | Configuration |
|---|---|
| Data source | gridMET daily precipitation |
| Time period | January 1990-December 2024 |
| Region | U.S. Southwest bounding box, approx. 31-37°N, 115-107°W |
| Input channels | SPI-3, SPI-6, SPI-12 |
| Forecast target | Next-month SPI field |
| Input sequence length | Six monthly fields |
| Split | Chronological 70% train, 15% validation, 15% test |
| Evaluation channel | SPI-3 |
| Main comparison | ConvLSTM versus shared per-pixel LSTM |
| Baselines | Persistence and climatology |

A chronological split is used to avoid leakage across time. Standardization is fit only on the training period and then applied to validation and test periods. Cells outside valid gridMET coverage inside the selected bounding box are zero-filled after preprocessing; this produces a visible flat region in the forecast-panel figure and is treated as a limitation rather than hidden.

## 3. Methods

### 3.1 Forecasting task

For each sample, the input is a six-month sequence of three-channel SPI fields over an H x W grid. The model predicts the next month's three-channel SPI field. Training loss is mean squared error over all three SPI channels, but the reported test metrics are computed on SPI-3 because the evaluation scripts use channel 0 for headline verification and drought-onset scoring.

### 3.2 ConvLSTM model

The ConvLSTM model follows the idea of Shi et al. (2015): recurrent gates are implemented with convolutional operations, so the hidden state remains a spatial feature map rather than a single vector. In this project, the configured model uses stacked ConvLSTM layers with 3 x 3 kernels and 48 hidden channels, followed by a 1 x 1 convolution to map the final hidden representation back to the three SPI channels. The relevant advantage is structural: neighboring cells can influence a grid cell's prediction through the convolutional gates.

### 3.3 Per-pixel LSTM baseline

The per-pixel LSTM is a control model designed to remove spatial coupling while keeping sequence modeling. The same LSTM weights are applied independently to each grid cell's six-month SPI history. It can learn temporal behavior common across the domain, but it cannot use nearby cells. This makes it a useful comparison because the main architectural difference is whether spatial neighborhoods are available.

### 3.4 Baselines

Persistence predicts that next month's field equals the most recent observed field. Climatology predicts the training-period mean field for every future month. These are intentionally simple baselines: a useful learning model should clear both.

### 3.5 Metrics

Evaluation uses RMSE, MAE, skill score versus persistence, anomaly correlation, drought-onset F1, and drought-onset recall. Drought onset is defined as a threshold crossing: the previous observed SPI-3 is at or above -0.8, and the current actual or predicted SPI-3 falls below -0.8. Persistence has zero onset F1 by construction under this definition because it copies the previous month and therefore cannot newly cross the threshold.

## 4. Results

### 4.1 One-month-ahead forecast metrics

**Table 2. One-month-ahead test metrics on SPI-3.**

| Model | RMSE | MAE | Skill vs. Persistence | Anomaly Corr. | Onset F1 | Onset Recall |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.826 | 0.570 | 0.000 | 0.739 | 0.000 | 0.000 |
| Climatology | 1.159 | 0.930 | -0.969 | 0.047 | 0.000 | 0.000 |
| Pixel-LSTM | **0.694** | **0.513** | **0.295** | **0.802** | 0.216 | 0.141 |
| ConvLSTM | 0.715 | 0.533 | 0.250 | 0.783 | **0.231** | **0.159** |

Both learned models outperform persistence and climatology on bulk error metrics. The per-pixel LSTM has the lowest RMSE and MAE, so the experiment does not support a broad claim that ConvLSTM is better for reconstructing the next SPI-3 field. The ConvLSTM's advantage appears in drought-onset detection: it has higher onset F1 and recall than the per-pixel LSTM. The absolute onset scores are still modest, which means most onset transitions remain difficult, but the spatial model detects a larger share of them.

### 4.2 Lead-time behavior

![Forecast skill and drought-onset F1 by lead time.](../assets/figures/leadtime.png)

**Figure 1.** Forecast skill and drought-onset F1 during autoregressive 1-6 month rollout. The left panel shows skill versus persistence; the right panel shows drought-onset F1. The models are similar on skill score, but ConvLSTM maintains higher onset F1 at every lead.

The lead-time experiment rolls both learned models forward autoregressively from one to six months. Skill versus persistence increases with lead time for both learned models, partly because the persistence reference weakens as the horizon grows. The more relevant difference is on onset F1: ConvLSTM remains above the per-pixel LSTM at every lead and the gap is largest at the six-month lead.

**Table 3. Drought-onset F1 by autoregressive lead time.**

| Lead (months) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| ConvLSTM | 0.217 | 0.103 | 0.088 | 0.133 | 0.136 | **0.148** |
| Pixel-LSTM | 0.188 | 0.093 | 0.040 | 0.038 | 0.045 | **0.056** |

This result should be interpreted carefully. It suggests that spatial context helps preserve some signal about newly emerging drought across rollout steps, but it does not prove that the mechanism is spatial propagation. The experiment is a controlled portfolio project and would need additional splits, regions, masks, and covariates before making stronger hydrologic claims.

### 4.3 Forecast field visualization

![Actual SPI-3 field compared with ConvLSTM and persistence forecasts.](../assets/figures/panels.png)

**Figure 2.** Actual SPI-3 fields, ConvLSTM one-month forecasts, and persistence forecasts at selected test months. Brown indicates drier conditions and green/blue indicates wetter conditions. The ConvLSTM fields are smoother than the observed and persistence fields, while persistence preserves the previous month's local texture.

The visualization supports the metric results. ConvLSTM produces smoother, spatially coherent fields, which can help around threshold boundaries but may also reduce local extremes. This helps explain why ConvLSTM can improve onset detection while not winning on RMSE. The flat wedge in the lower-left portion of some panels is a preprocessing artifact from cells inside the bounding box but outside valid gridMET CONUS coverage. A stricter mask would be a necessary next step.

## 5. Discussion

The main result is metric-dependent. Bulk field reconstruction favors the per-pixel LSTM, while onset detection favors the ConvLSTM. This is plausible because RMSE is dominated by all pixels, including many that are not near the drought-onset boundary. A model can achieve lower average error by following each pixel's autocorrelation closely. Onset detection is a harder boundary-crossing problem: small changes near the -0.8 threshold matter more than small errors elsewhere. Local spatial context can stabilize those boundary predictions.

The lead-time results strengthen this interpretation but do not make it conclusive. The ConvLSTM's onset F1 remains higher across every tested lead, especially after lead 3, where the per-pixel LSTM's onset F1 collapses. Still, the experiment uses one domain, one split, one random seed, and SPI-only inputs. These conditions are sufficient for a reproducible baseline study, but not for claiming operational drought-forecasting performance.

A practical takeaway is that different metrics answer different questions. RMSE asks how closely the model reconstructs the next field on average. Onset F1 asks whether the model identifies new drought emergence. For early-warning use, onset metrics are more directly aligned with the decision problem, but they should be paired with precision, recall, spatial masks, uncertainty estimates, and independent drought labels in future work.

## 6. Limitations and Validity Checks

- The Southwest bounding box includes cells outside valid gridMET CONUS coverage; these are zero-filled and visible in the figures.
- The reported metrics are based on one chronological split and one random seed, so small differences should not be overinterpreted.
- The evaluation focuses on SPI-3 even though the models also predict SPI-6 and SPI-12.
- The input set is precipitation-only; no ENSO, temperature, soil moisture, evapotranspiration, vegetation, or land-surface predictors are included.
- Monthly aggregation and SPI accumulation smooth short-timescale drought dynamics.
- No uncertainty quantification or ensemble forecasting is included.
- Region-holdout code exists in the repository, but region-holdout results are not reported here.
- The model is not validated against an independent drought reference such as the U.S. Drought Monitor.
- This work should be treated as a reproducible research prototype, not an operational drought-monitoring system.

## 7. Reproducibility

The repository is organized around a minimal end-to-end workflow:

```text
src/
  data.py            gridMET download, SPI construction, windowing, chronological split
  models.py          ConvLSTM forecaster and per-pixel LSTM baseline
  train.py           model training and checkpoint selection
  evaluate.py        one-month metrics and baseline comparison
  leadtime.py        autoregressive 1-6 month lead-time experiment
  visualize.py       forecast panels and animation
  synth.py           synthetic drought-field generator for pipeline testing
  region_holdout.py  spatial generalization experiment, not reported here
config.yaml          real-data Southwest configuration
config_demo.yaml     fast synthetic demo configuration
results/             metrics and lead-time JSON outputs
assets/figures/      figures used in the report and README
```

Real-data run:

```bash
pip install -r requirements.txt
python src/train.py     --config config.yaml
python src/evaluate.py  --config config.yaml
python src/leadtime.py  --config config.yaml --max_lead 6
python src/visualize.py --config config.yaml
```

Synthetic pipeline check:

```bash
python src/train.py    --config config_demo.yaml
python src/evaluate.py --config config_demo.yaml
```

The synthetic pipeline makes it possible to test the code path without downloading climate data. The real-data path requires network access to the gridMET THREDDS server and the required Python packages listed in `requirements.txt`.

## 8. Conclusion

This project shows a clear but specific result: spatial modeling does not automatically improve every drought-forecast metric. In the one-month test, the per-pixel LSTM performs slightly better on RMSE and MAE, while the ConvLSTM performs better on drought-onset F1 and recall. In 1-6 month autoregressive rollouts, ConvLSTM retains higher onset F1 at every lead. The result suggests that spatial context is most useful when the forecasting target is newly emerging drought rather than average field reconstruction. Future work should add spatial masks, repeated seeds, region-holdout tests, additional predictors, and independent validation before treating the model as a stronger drought-forecasting system.

## References

Abatzoglou, J. T. (2013). Development of gridded surface meteorological data for ecological applications and modelling. *International Journal of Climatology*, 33(1), 121-131.

McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *Proceedings of the 8th Conference on Applied Climatology*, American Meteorological Society.

Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., & Woo, W. (2015). Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. *Advances in Neural Information Processing Systems*.

National Integrated Drought Information System. `climate_indices` Python package, used in this project to compute SPI from precipitation.

---

Independent project, Arizona State University. Real-data training and evaluation were run on ASU Research Computing's Sol supercomputer. Code, configuration, results, and report artifacts are maintained in the project repository.
