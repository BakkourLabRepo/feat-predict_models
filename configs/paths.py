"""
Central data paths used by all configs.

By default, data live in a `data/` directory at the root of this repository:

    data/
    ├── human/
    │   ├── exp_1/{data,results}
    │   ├── exp_2/{data,results}
    │   └── exp_mri/{beh,derivatives/beh}
    └── simulations/
        └── exp-<N>/data/...

To keep data elsewhere, set the FEAT_PREDICT_DATA environment variable, e.g.

    export FEAT_PREDICT_DATA=/path/to/data
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get('FEAT_PREDICT_DATA') or os.path.join(REPO_ROOT, 'data')
))

HUMAN_PATH = os.path.join(DATA_DIR, 'human')
SIM_PATH = os.path.join(DATA_DIR, 'simulations')
