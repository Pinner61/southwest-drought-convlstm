# v1.2 Upgrade Notes

This upgrade adds five portfolio-strength improvements:

1. Region holdout experiment (`src/region_holdout.py`)
2. Invalid-cell masking for gridMET edge artifacts (`src/train.py`, `src/evaluate.py`, `src/leadtime.py`, `src/visualize.py`)
3. Separate SPI-3, SPI-6, and SPI-12 evaluation (`results/metrics_by_channel.json`)
4. Training/validation loss figure (`src/plot_training_curves.py`)
5. Streamlit dashboard (`app/dashboard.py`)

## Run order

```bash
python src/train.py --config config.yaml
python src/evaluate.py --config config.yaml
python src/leadtime.py --config config.yaml --max_lead 6
python src/visualize.py --config config.yaml
python src/plot_training_curves.py --config config.yaml
python src/region_holdout.py --config config.yaml
streamlit run app/dashboard.py -- --out outputs_real
```

For a quick local check, use `config_demo.yaml` instead of `config.yaml`.
