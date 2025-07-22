import numpy as np

PROJECT_PATH = ''

AGENT_CONFIGS_PATH = False

experiment_config = {
    
    'depth-1_dim-2': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-3/data',

        # Random seed for reproducibility
        'seed': 5423,

        # Number of training trials
        'n_training_target_repeats': 6,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 1000,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

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


        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,1,0,0],
                [1,4,0,0],
                [1,4,0,0],
                [4,1,0,0],
                [4,1,0,0],
                [4,4,0,0],
                [0,0,1,1],
                [0,0,1,4],
                [0,0,1,4],
                [0,0,4,1],
                [0,0,4,1],
                [0,0,4,4]
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,1,0,0],
            [1,0,1,0],
            [1,0,0,1],
            [0,1,1,0],
            [0,1,0,1],
            [0,0,1,1],
        ]),


        # Environment config
        'env_config': {
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
            'r': np.array([[-1,0,0,1]]),
            'continuous_features': False
        },

    },

}