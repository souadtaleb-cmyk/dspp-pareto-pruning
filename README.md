# DSPP — Dynamic Streaming Pareto Pruning

Code accompanying the manuscript *"Dynamic Streaming Pareto Pruning for
Resource-Efficient Ensemble Classification on Evolving Data Streams"*,
submitted to the International Journal of Machine Learning and Cybernetics.

This repository contains the full experimental pipeline: the DSPP method
(dynamic and static variants), all ablations, all baselines, the eight
evaluated data streams, and the statistical analysis (Friedman test, paired
Wilcoxon tests) used to produce Tables 1–4 and Figures 1–2 in the manuscript.

## Repository structure

```
.
├── src/
│   ├── dspp_core.py       # DSPPRunner (Pareto front, knee-point selection,
│   │                       #   drift-triggered pool adaptation), baselines
│   └── experiment.py       # Stream registry, run_all orchestration,
│                            #   result tables, statistical tests
├── notebooks/
│   └── DSPP_pipeline_extended.ipynb   # Self-contained Colab notebook
│                                       #   (installs deps, runs everything,
│                                       #   exports tables + figure)
├── figures/
│   ├── make_kappa_figure.py           # Reproduces Figure 1 (Table 1bis)
│   ├── make_efficiency_figure.py      # Reproduces Figure 2 (Table 4)
│   ├── kappa_creditcard_http.pdf
│   └── accuracy_vs_memory.pdf
├── results/
│   ├── dspp_raw_results.csv           # Raw results (800 rows: 8 streams
│   │                                   #   x 10 methods x 10 seeds) used to
│   │                                   #   produce every table/figure in the
│   │                                   #   manuscript — see "Reproducing the
│   │                                   #   tables without re-running the
│   │                                   #   pipeline" below
│   └── dspp_result_tables.xlsx        # Tables 1, 1bis, 2, 3, 4 + Friedman
│                                       #   test, generated from the CSV above
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- `river` 0.25.0 (note: requires `numpy>=2.3.4, scipy>=1.16` — on Google
  Colab, this conflicts with the pre-installed numpy/scipy binaries; see
  "Running on Google Colab" below)

Install with:

```bash
pip install -r requirements.txt
```

## Reproducing the tables without re-running the pipeline

`results/dspp_raw_results.csv` contains the raw output of the full 10-seed,
8-stream, 10-method run reported in the manuscript (800 rows). All tables
and figures can be regenerated from this file directly, without re-running
the ~7.5-hour pipeline:

```python
import pandas as pd
from src.experiment import table1_accuracy, table1_kappa, friedman_test, \
    table2_paired_tests, table3_per_stream_gain, table4_efficiency

df = pd.read_csv('results/dspp_raw_results.csv')

table1_accuracy(df)   # Table 1
table1_kappa(df)      # Table 1bis
friedman_test(df, ['DSPP-Lite', 'DSPP-Adaptive', 'StaticSPP-Lite',
                    'OB', 'LB', 'ARF', 'SRP'])
table2_paired_tests(df)               # Table 2 (Wilcoxon)
table3_per_stream_gain(df, 'Lite')    # Table 3
table4_efficiency(df)                 # Table 4
```

`results/dspp_result_tables.xlsx` contains these same tables pre-computed,
one sheet per table.

## Running the pipeline

### Locally / on a server

```python
from src.experiment import run_all, table1_accuracy, table1_kappa, \
    friedman_test, table2_paired_tests, table3_per_stream_gain, table4_efficiency

results_df = run_all(seeds=range(10))   # all 8 streams, full instance counts
results_df.to_csv('results/dspp_raw_results.csv', index=False)

print(table1_accuracy(results_df))
print(table1_kappa(results_df))
```

A full run (10 seeds × 8 streams × 10 methods, including HTTP at its full
567,498 instances) took approximately 7.5 hours on Google Colab's free CPU
runtime. `CreditCard` and `HTTP` trigger an automatic one-time download via
`river` (~66 MB and ~4 MB respectively) on first access.

### On Google Colab

Open `notebooks/DSPP_pipeline_extended.ipynb` directly in Colab. The
notebook is self-contained (installs `river`, defines the full pipeline,
runs it, and exports results/figures) and includes a `QUICK_TEST` flag for
a fast smoke-test run before committing to the full-scale experiment.

**Known Colab issue:** installing `river` upgrades `numpy`/`scipy` to
versions incompatible with the pre-loaded binaries in a running Colab
kernel. The first code cell must therefore run:

```python
!pip install -U numpy scipy river --quiet
import os
os.kill(os.getpid(), 9)  # forces a clean kernel restart to reload the new binaries
```

before any other cell — otherwise `from scipy import stats` will raise
`ImportError: cannot import name '_center' from 'numpy._core.umath'`.

### Reproducing the figures only

```bash
cd figures
python make_kappa_figure.py        # -> kappa_creditcard_http.pdf
python make_efficiency_figure.py   # -> accuracy_vs_memory.pdf
```

Both scripts embed the aggregated values from Tables 1bis and 4 directly
(no need to re-run the full pipeline first) for quick regeneration during
manuscript revisions.

## Reproducibility notes

- All results in the manuscript use 10 seeds (`range(10)`), `WINDOW_SIZE=200`,
  `GAMMA=0.01`, `EVICT_THRESHOLD=0.5`, `N_INJECT=2`, `MAX_POOL=12` — see
  `src/dspp_core.py` module-level constants.
- `CreditCard` is capped at the first 60,000 instances (163 fraud cases fall
  within this range); `HTTP` is run on its full 567,498 instances (its 2,211
  positive instances are concentrated between indices 201,669–514,439, so
  any smaller cap misses the positive class entirely — see Section 6.1 of
  the manuscript).
- Each DSPP base learner's internal RNG (used for per-model Poisson(1)
  online bagging in the Lite variant) is seeded as `100 * seed + model_id`
  for full reproducibility across pool-adaptation events.
- Memory figures are measured as serialized (`pickle`) model size in KB, a
  standard proxy rather than live runtime memory profiling — noted as a
  methodological choice in Section 5.6 of the manuscript.

## Citation

If you use this code, please cite:

```bibtex
@article{talebzouggar2026dspp,
  title   = {Dynamic Streaming Pareto Pruning for Resource-Efficient
             Ensemble Classification on Evolving Data Streams},
  author  = {Taleb Zouggar, Souad},
  journal = {International Journal of Machine Learning and Cybernetics},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT License — see `LICENSE`.

## Contact

Souad Taleb Zouggar — talebzouggar.souad@univ-oran2.dz
Department of Economic Science, University of Oran 2, Algeria
