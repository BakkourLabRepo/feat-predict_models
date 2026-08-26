import numpy as np

experiment_config = {

    # Choice data path
    'data_path': None,

    # Results path and file name
    'results_path': None,
    'results_fname': 'model_fits',

    # Whether data is in BIDS format
    'bids': False,

    # Participant IDs to exclude
    'ids_to_exclude': [],

    # Optimizer settings
    'n_starts': 100, # Max number of random starts
    'max_unchanged': 5, # Max number of random starts without improvement

    # Overwrite existing results
    'overwrite': True,

    # Number of cores to use
    'num_cores': 4,

    # Configurations for models to fit
    'model_configs': [

        # Feature -> Feature Successor Feature model
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ff',
                'alpha': None, 
                'beta': None,
            }
        ),

        # Feature -> Feature Successor Feature model (w/ bias)
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ff-bias',
                'alpha': None, 
                'beta': None,
                'bias_magnitude': None,
            }
        ),

        # Feature -> Feature Successor Feature model (w/ decay)
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ff-decay',
                'alpha': None, 
                'beta': None,
                'lmbd': None,
            }
        ),

        # Feature -> Feature Successor Feature model (w/ regularization)
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ff-l1reg',
                'alpha': None, 
                'beta': None,
                'lmbd_l1': None,
            }
        ),

        # Feature -> Feature Successor Feature model
        # (w/ non-linear value estimation)
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ff-nonlin',
                'alpha': None, 
                'beta': None,
                'inference_power': None,
            }
        ),

        # Feature -> Feature Successor Feature model (w/ bias & decay)
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ff-bias-decay',
                'alpha': None, 
                'beta': None,
                'lmbd': None,
                'bias_magnitude': None,
            }
        ),

        # Feature -> Feature Successor Feature model
        # (w/ bias & regularization)
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ff-bias-l1reg',
                'alpha': None, 
                'beta': None,
                'lmbd_l1': None,
                'bias_magnitude': None,
            }
        ),

        # Feature -> Feature Successor Feature model
        # (w/ bias & non-linear value estimation)
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ff-bias-nonlin',
                'alpha': None, 
                'beta': None,
                'bias_magnitude': None,
                'inference_power': None,
            }
        ),

        # State -> State model Successor Features model
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ss',
                'alpha': None, 
                'beta': None,
                'conjunctive_starts': True,
                'conjunctive_successors': True,
                'conjunctive_composition': True,
            }
        ),

        # State -> State Sampler Successor Features model
        (
            'SuccessorFeatures',
            {
                'id': None,
                'model_label': 'sf-ss-sampler',
                'alpha': None, 
                'beta': None,
                'conjunctive_starts': True,
                'conjunctive_successors': True,
                'memory_sampler': True,
                'sampler_feature_weight': None,
                'sampler_recency_weight': 0,
                'sampler_specificity': None
            }
        ),

        # Feature -> Feature MBRL model
        (
            'MBRL',
            {
                'id': None,
                'model_label': 'mb-ff',
                'alpha': None, 
                'beta': None,
            }
        ),

        # Feature -> Feature MBRL model (w/ bias)
        (
            'MBRL',
            {
                'id': None,
                'model_label': 'mb-ff-bias',
                'alpha': None, 
                'beta': None,
                'bias_magnitude': None,
            }
        ),

        # State -> State model MBRL model
        (
            'MBRL',
            {
                'id': None,
                'model_label': 'mb-ss',
                'alpha': None, 
                'beta': None,
                'conjunctive_starts': True,
                'conjunctive_successors': True,
                'conjunctive_composition': True,
            }
        ),

        # State -> State Sampler MBRL model
        (
            'MBRL',
            {
                'id': None,
                'model_label': 'mb-ss-sampler',
                'alpha': None, 
                'beta': None,
                'conjunctive_starts': True,
                'conjunctive_successors': True,
                'memory_sampler': True,
                'sampler_feature_weight': None,
                'sampler_recency_weight': 0,
                'sampler_specificity': None
            }
        )

        ],

    # Parameter bounds
    'parameter_bounds': {
        'alpha': (0, 1),
        'beta': (.0001, np.inf),
        'lmbd': (0, 1),
        'lmbd_l1': (0, np.inf),
        'bias_magnitude': (0, 1),
        'inference_power': (0, np.inf),
        'sampler_feature_weight': (0, 1),
        'sampler_recency_weight': (0, 1),
        'sampler_specificity': (1, np.inf)
    },

    # Environment config
    'env_config': {
        'tmat': np.array([
            [1,0,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,0,1]
        ]),
        'max_steps': 1,
        'n_feats': 4,
        'n_fixed': 0,
        'n_per': 2,
        'start_insts': np.array([[2, 3]]),
        'r': np.array([[-1, 0, 0, 1]]),
        'continuous_features': False
    },

    # How features reorder at each step in each condition
    'feature_reorder': []
}
