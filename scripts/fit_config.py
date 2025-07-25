import numpy as np

# Choice data path
DATA_PATH = '/Users/euanprentis/Documents/feat_predict_simulations/exp-3/data/ff'

# Results path and file name
RESULTS_PATH = '/Users/euanprentis/Documents/feat_predict_simulations/exp-3/results'
RESULTS_FNAME = 'model_fits'

# Optimizer settings
N_STARTS = 100 # Max number of random starts
MAX_UNCHANGED = 5 # Max number of random starts without improvement

# Overwrite existing results
OVERWRITE = True

# Number of cores to use
NUM_CORES = 4

# Configurations for models to fit
MODEL_CONFIGS = [

    # Feature -> Feature model
    {
        'id': None,
        'model_label': 'ff',
        'alpha': None, 
        'alpha_2': 'alpha',
        'alpha_decay': 0, 
        'beta': None,
        'beta_test': 'beta',
        'gamma': 1.,
        'segmentation': None,
        'segmentation_2': 'segmentation',
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
        'alpha_2': 'alpha',
        'alpha_decay': 0, 
        'beta': None,
        'beta_test': 'beta',
        'gamma': 1.,
        'segmentation': 0,
        'segmentation_2': 'segmentation',
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
        'alpha_2': 'alpha',
        'alpha_decay': 0, 
        'beta': None,
        'beta_test': 'beta',
        'gamma': 1.,
        'segmentation': 0,
        'segmentation_2': 'segmentation',
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
    'alpha': (.0001, .9999),
    'alpha_2': (.0001, .9999),
    'alpha_decay': (0, np.inf), 
    'beta': (.0001, np.inf),
    'beta_test': (.0001, np.inf),
    'segmentation': (-.9999, .9999),
    'segmentation_2': (-.9999, .9999),
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



