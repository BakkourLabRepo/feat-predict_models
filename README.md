# feat-predict_models

Code for simulating and fitting feature-prediction learning models: Successor
Features (`src/SuccessorFeatures.py`) and model-based RL (`src/MBRL.py`).

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate feat-predict-models
```

## Data

All configs read and write data under a single data directory. By default this
is `data/` at the root of the repository (this folder is ignored by git). To use
a different location, set the `FEAT_PREDICT_DATA` environment variable:

```bash
export FEAT_PREDICT_DATA=/path/to/data
```

Paths are defined in [`configs/paths.py`](configs/paths.py). The expected layout is:

```
data/
├── human/
│   ├── exp_1/
│   │   ├── data/          # training/ and test/ CSVs, one file per participant
│   │   └── results/       # model fits (written by fit.py)
│   ├── exp_2/
│   │   ├── data/
│   │   └── results/
│   └── exp_mri/
│       ├── beh/           # BIDS-formatted behavioural data (sub-*/)
│       └── derivatives/
│           └── beh/       # model fits (written by fit.py)
└── simulations/
    └── exp-<N>/data/      # simulated agents (written by simulate.py)
```

## Usage

Run all scripts from the repository root.

### Simulate agents

```bash
python simulate.py <config>
```

where `<config>` is the name of a file in `configs/simulate/` (e.g.
`python simulate.py exp-1`). Each config contains one or more simulation
conditions. Results are saved to `data/simulations/<config>/data/<condition>/`.

### Fit models to data

```bash
python fit.py <config> [--dpath DATA_PATH] [--rpath RESULTS_PATH]
```

where `<config>` is the name of a file in `configs/fit/` (e.g.
`python fit.py exp_1`). `--dpath` and `--rpath` override the data and results
paths set in the config. The `*_recovery` configs have no default paths and
need both to be passed.
