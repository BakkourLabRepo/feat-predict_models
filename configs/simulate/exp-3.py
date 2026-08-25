import numpy as np

PROJECT_PATH = '/Users/euanprentis/Documents/feat_predict_simulations'

AGENT_CONFIGS_PATH = False

HUMAN_PATH = '/Users/euanprentis/Library/CloudStorage/Box-Box/Bakkour-Lab/projects/feat-predict/human'

experiment_config = {
    
    'exp_1': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-3/data/exp_1',

        # Number of training trials
        'n_training_target_repeats': 6,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': f'{HUMAN_PATH}/exp_1/data/training',
        'test_trial_info_path': f'{HUMAN_PATH}/exp_1/data/test',
        'match_trials_to_agents': False,

        # How features reorder at each step in each condition
        # (only relevant when loading human experiment data)
        'feature_reorder': np.array([
    
            # Semantic congruent (only 1 step)
            [[0,1,2,3]],
    
            # Semantic incongruent (only 1 step)
            [[2,3,0,1]]
    
        ]),

        # Number of agents per basic agent config
        'n_agents': 1000,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
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


    'exp_2': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-3/data/exp_2',

        # Number of training trials
        'n_training_target_repeats': 6,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': f'{HUMAN_PATH}/exp_2/data/training',
        'test_trial_info_path': f'{HUMAN_PATH}/exp_2/data/test',
        'match_trials_to_agents': False,

        # How features reorder at each step in each condition
        # (only relevant when loading human experiment data)
        'feature_reorder': np.array([

            # Condition 1 (1I) - semantic congruent
            [

                # Step 1
                [0,1,2,3],

                # Step 2
                [0,1,2,3]

            ],

            # Condition 2 (1II) - semantic incongruent

            [
                # Step 1
                [3,2,1,0],

                # Step 2
                [2,3,0,1]

            ],

            # Condition 3 (2I) - semantic incongruent
            [

                # Step 1
                [2,3,0,1],

                # Step 2
                [3,2,1,0]

            ],

            # Condition 4 (3II) - semantic congruent
            [

                # Step 1
                [0,1,2,3],

                # Step 2
                [0,1,2,3]

            ],

            # Condition 5 (2II) - semantic congruent
            [

                # Step 1
                [0,1,2,3],

                # Step 2
                [0,1,2,3]

            ],

            # Condition 6 (3II) - semantic incongruent
            [
                # Step 1
                [2,3,0,1],

                # Step 2
                [3,2,1,0]

            ]
            
        ]),

        # Number of agents per basic agent config
        'n_agents': 1000,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
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
                [1,0,0,0,0,0],
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1],
                [0,0,0,0,0,1]
            ]),
            'max_steps': 1,
            'n_feats': 4,
            'n_fixed': 0,
            'n_per': 2,
            'start_insts': np.array([[3, 4], [2, 5]]),
            'r': np.array([[-1, 0, 0, 0, 0, 1]]),
            'continuous_features': False
        }

    },

}