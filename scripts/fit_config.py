import numpy as np

# Choice data path
DATA_PATH = '/Users/euanprentis/Documents/feat_predict_simulations/data/ss-sampler'
DATA_PATH = '/Users/euanprentis/Library/CloudStorage/Box-Box/Bakkour-Lab/users/euan/feat-predict/data/human/exp_2'

# Results path
RESULTS_PATH = '/Users/euanprentis/Documents/feat_predict_simulations/results'
RESULTS_PATH = '/Users/euanprentis/Library/CloudStorage/Box-Box/Bakkour-Lab/users/euan/feat-predict/results/human/exp_2'

# Optimizer settings
N_STARTS = 100 # Max number of random starts
MAX_UNCHANGED = 5 # Max number of random starts without improvement

# Overwrite existing results
OVERWRITE = False

# Number of cores to use
NUM_CORES = 4

# Configurations for models to fit
MODEL_CONFIGS = [

    # Feature -> Feature model
    {
        'id': None,
        'model_label': 'ff',
        'alpha': None, 
        'beta': None,
        'beta_test': 'beta',
        'gamma': 1.,
        'segmentation': None,
        'conjunctive_starts': False,
        'conjunctive_successors': False,
        'conjunctive_composition': False,
        'memory_sampler': False,
        'sampler_feature_weight': 1,
        'sampler_recency_weight': 0,
        'sampler_specificity': 1
    },

    # State -> State model
    {
        'id': None,
        'model_label': 'ss',
        'alpha': None, 
        'beta': None,
        'beta_test': 'beta',
        'gamma': 1.,
        'segmentation': 1,
        'conjunctive_starts': True,
        'conjunctive_successors': True,
        'conjunctive_composition': True,
        'memory_sampler': False,
        'sampler_feature_weight': 1,
        'sampler_recency_weight': 0,
        'sampler_specificity': 1
    },

    # State -> State Sampler model
    {
        'id': None,
        'model_label': 'ss-sampler',
        'alpha': None, 
        'beta': None,
        'beta_test': 'beta',
        'gamma': 1.,
        'segmentation': 1,
        'conjunctive_starts': True,
        'conjunctive_successors': True,
        'conjunctive_composition': False,
        'memory_sampler': True,
        'sampler_feature_weight': None,
        'sampler_recency_weight': 0,
        'sampler_specificity': None
    }

]

# Parameter bounds
PARAMETER_BOUNDS = {
    'alpha': (.0001, 1),
    'beta': (.0001, np.inf),
    'beta_test': (.0001, np.inf),
    'segmentation': (0, .9999),
    'sampler_feature_weight': (0, 1),
    'sampler_recency_weight': (0, 1),
    'sampler_specificity': (1, np.inf)
}

# Environment config
ENV_CONFIG = {
    'tmat': np.array([
        [1,0,0,0],
        [1,0,0,0],
        [0,0,0,1],
        [0,0,0,1]
    ]),
    'n_feats': 4,
    'n_fixed': 0,
    'n_per': 2,
    'start_insts': np.array([2, 3]),
    'r': np.array([[-1, 0, 0, 1]]),
    'continuous_features': False
}

# How features reorder for the between-feature transitions condition
FEATURE_REORDER = np.array([2,3,0,1])



