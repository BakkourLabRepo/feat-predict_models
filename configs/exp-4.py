import numpy as np

PROJECT_PATH = '/Users/euanprentis/Documents/feat_predict_simulations_2'

AGENT_CONFIGS_PATH = False

experiment_config = {

    'depth-1_dim-1': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-1_dim-1',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90*3, 

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,0,0,0],
                [4,0,0,0],
                [0,1,0,0],
                [0,4,0,0],
                [0,0,1,0],
                [0,0,4,0],
                [0,0,0,1],
                [0,0,0,4],
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
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
            'n_per': 1,
            'start_insts': np.array([2, 3]),
            'r': np.array([[-1,0,0,1]]),
            'continuous_features': False
        },

    },
    
    'depth-1_dim-2': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-1_dim-2',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff-biased',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 1,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

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


    'depth-2_dim-1': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-2_dim-1',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90*3, 

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,0,0,0],
                [6,0,0,0],
                [0,1,0,0],
                [0,6,0,0],
                [0,0,1,0],
                [0,0,6,0],
                [0,0,0,1],
                [0,0,0,6],
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
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
            'n_feats': 4,
            'n_fixed': 0,
            'n_per': 1,
            'start_insts': np.array([3, 4]),
            'r': np.array([[-1,0,0,0,0,1]]),
            'continuous_features': False
        },

    },


    'depth-2_dim-2': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-2_dim-2',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff-biased',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 1,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            )

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,1,0,0],
                [1,6,0,0],
                [1,6,0,0],
                [6,1,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [0,0,1,1],
                [0,0,1,6],
                [0,0,1,6],
                [0,0,6,1],
                [0,0,6,1],
                [0,0,6,6]
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
            'n_feats': 4,
            'n_fixed': 0,
            'n_per': 2,
            'start_insts': np.array([3, 4]),
            'r': np.array([[-1,0,0,0,0,1]]),
            'continuous_features': False
        },

    },



    'depth-2_dim-4': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-2_dim-4',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff-biased',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 1,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            )

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,1,1,1,0,0,0,0],
                [1,1,6,6,0,0,0,0],
                [1,1,6,6,0,0,0,0],
                [6,6,1,1,0,0,0,0],
                [6,6,1,1,0,0,0,0],
                [6,6,6,6,0,0,0,0],
                [0,0,0,0,1,1,1,1],
                [0,0,0,0,1,1,6,6],
                [0,0,0,0,1,1,6,6],
                [0,0,0,0,6,6,1,1],
                [0,0,0,0,6,6,1,1],
                [0,0,0,0,6,6,6,6]
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,1,1,1,0,0,0,0],
            [1,1,0,0,1,1,0,0],
            [1,1,0,0,0,0,1,1],
            [0,0,1,1,1,1,0,0],
            [0,0,1,1,0,0,1,1],
            [0,0,0,0,1,1,1,1],
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
            'n_feats': 8,
            'n_fixed': 0,
            'n_per': 4,
            'start_insts': np.array([3, 4]),
            'r': np.array([[-1,0,0,0,0,1]]),
            'continuous_features': False
        },

    },



    'depth-3_dim-1': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-3_dim-1',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90*3, 

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,0,0,0],
                [8,0,0,0],
                [0,1,0,0],
                [0,8,0,0],
                [0,0,1,0],
                [0,0,8,0],
                [0,0,0,1],
                [0,0,0,8],
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]),


        # Environment config
        'env_config': {
            'tmat': np.array([
                [1,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,1]
            ]),
            'n_feats': 4,
            'n_fixed': 0,
            'n_per': 1,
            'start_insts': np.array([4, 5]),
            'r': np.array([[-1,0,0,0,0,0,0,1]]),
            'continuous_features': False
        },

    },


    'depth-3_dim-2': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-3_dim-2',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff-biased',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 1,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            )

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,1,0,0],
                [1,8,0,0],
                [1,8,0,0],
                [8,1,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [0,0,1,1],
                [0,0,1,8],
                [0,0,1,8],
                [0,0,8,1],
                [0,0,8,1],
                [0,0,8,8]
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
                [1,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,1]
            ]),
            'n_feats': 4,
            'n_fixed': 0,
            'n_per': 2,
            'start_insts': np.array([4, 5]),
            'r': np.array([[-1,0,0,0,0,0,0,1]]),
            'continuous_features': False
        },

    },


    'depth-3_dim-4': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-3_dim-4',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff-biased',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 1,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            )

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,1,1,1,0,0,0,0],
                [1,1,8,8,0,0,0,0],
                [1,1,8,8,0,0,0,0],
                [8,8,1,1,0,0,0,0],
                [8,8,1,1,0,0,0,0],
                [8,8,8,8,0,0,0,0],
                [0,0,0,0,1,1,1,1],
                [0,0,0,0,1,1,8,8],
                [0,0,0,0,1,1,8,8],
                [0,0,0,0,8,8,1,1],
                [0,0,0,0,8,8,1,1],
                [0,0,0,0,8,8,8,8]
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,1,1,1,0,0,0,0],
            [1,1,0,0,1,1,0,0],
            [1,1,0,0,0,0,1,1],
            [0,0,1,1,1,1,0,0],
            [0,0,1,1,0,0,1,1],
            [0,0,0,0,1,1,1,1],
        ]),


        # Environment config
        'env_config': {
            'tmat': np.array([
                [1,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,1]
            ]),
            'n_feats': 8,
            'n_fixed': 0,
            'n_per': 4,
            'start_insts': np.array([4, 5]),
            'r': np.array([[-1,0,0,0,0,0,0,1]]),
            'continuous_features': False
        },


    },

    
    'depth-4_dim-1': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-4_dim-1',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90*3, 

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,0,0,0],
                [10,0,0,0],
                [0,1,0,0],
                [0,10,0,0],
                [0,0,1,0],
                [0,0,10,0],
                [0,0,0,1],
                [0,0,0,10],
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]),


        # Environment config
        'env_config': {
            'tmat': np.array([
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,1]
            ]),
            'n_feats': 4,
            'n_fixed': 0,
            'n_per': 1,
            'start_insts': np.array([5, 6]),
            'r': np.array([[-1,0,0,0,0,0,0,0,0,1]]),
            'continuous_features': False
        },

    },


    'depth-4_dim-2': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-4_dim-2',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff-biased',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 1,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            )

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,1,0,0],
                [1,10,0,0],
                [1,10,0,0],
                [10,1,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [0,0,1,1],
                [0,0,1,10],
                [0,0,1,10],
                [0,0,10,1],
                [0,0,10,1],
                [0,0,10,10]
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
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,1]
            ]),
            'n_feats': 4,
            'n_fixed': 0,
            'n_per': 2,
            'start_insts': np.array([5, 6]),
            'r': np.array([[-1,0,0,0,0,0,0,0,0,1]]),
            'continuous_features': False
        },

    },



    'depth-4_dim-4': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-4/data/depth-4_dim-4',

        # Random seed for reproducibility
        'seed': 243423,

        # Number of training trials
        'n_training_target_repeats': 90,

        # Simulate based on existing agent configurations
        'agent_configs_path': AGENT_CONFIGS_PATH,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if AGENT_CONFIGS_PATH is not False
        'model_configs': [

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 0,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            ),

            (
                'MBRL',
                {
                    'id': None,
                    'model_label': 'ff-biased',
                    'alpha': None, 
                    'alpha_decay': 0, 
                    'beta': None,
                    'gamma': 1,
                    'bias_magnitude': 1,
                    'conjunctive_starts': False,
                    'conjunctive_successors': False,
                    'conjunctive_composition': False,
                    'memory_sampler': False,
                    'sampler_feature_weight': 1,
                    'sampler_recency_weight': 0,
                    'sampler_specificity': 1
                }
            )

        ],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,1,1,1,0,0,0,0],
                [1,1,10,10,0,0,0,0],
                [1,1,10,10,0,0,0,0],
                [10,10,1,1,0,0,0,0],
                [10,10,1,1,0,0,0,0],
                [10,10,10,10,0,0,0,0],
                [0,0,0,0,1,1,1,1],
                [0,0,0,0,1,1,10,10],
                [0,0,0,0,1,1,10,10],
                [0,0,0,0,10,10,1,1],
                [0,0,0,0,10,10,1,1],
                [0,0,0,0,10,10,10,10]
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,1,1,1,0,0,0,0],
            [1,1,0,0,1,1,0,0],
            [1,1,0,0,0,0,1,1],
            [0,0,1,1,1,1,0,0],
            [0,0,1,1,0,0,1,1],
            [0,0,0,0,1,1,1,1],
        ]),


        # Environment config
        'env_config': {
            'tmat': np.array([
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,1]
            ]),
            'n_feats': 8,
            'n_fixed': 0,
            'n_per': 4,
            'start_insts': np.array([5, 6]),
            'r': np.array([[-1,0,0,0,0,0,0,0,0,1]]),
            'continuous_features': False
        },


    },

}