import numpy as np

# Output directory
OUTPUT_PATH = '/Users/euanprentis/Documents/feat_predict_simulations/data'

# Random seed for reproducibility
SEED = 3244343

# Number of training trials
N_TRAINING_TARGET_REPEATS = 6

# Simulate based on existing agent configurations
LOAD_AGENT_CONFIGS = False
AGENT_CONFIGS_PATH = '/Users/euanprentis/Library/CloudStorage/Box-Box/Bakkour-Lab/users/euan/feat-predict/results/human/exp_2/fit_agent_configs'

# Number of agents per basic agent config
# Only need to set if LOAD_AGENT_CONFIGS = False
N_AGENTS = 1000

# Configurations for models to simulate
# Only need to set if LOAD_AGENT_CONFIGS = False
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

# Training targets
TRAINING_TARGETS_SET = np.array([

    # Block 1
    [
        [1,1,0,0],
        [1,4,0,0],
        [1,4,0,0],
        [4,1,0,0],
        [4,1,0,0],
        [4,4,0,0],
    ],

    # Block 2
    [
        [0,0,1,1],
        [0,0,1,4],
        [0,0,1,4],
        [0,0,4,1],
        [0,0,4,1],
        [0,0,4,4]
    ]

])

# Test feature combinations in the composition set
TEST_COMBS_SET = np.array([
    [1,1,0,0],
    [1,0,1,0],
    [1,0,0,1],
    [0,1,1,0],
    [0,1,0,1],
    [0,0,1,1],
])


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
