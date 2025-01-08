import numpy as np

experiment_config = {
    
    'model-ff_bias-none_sdim-2_depth-1_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-2/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 90,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

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
                'model_label': 'model-ff_bias-none_sdim-2_depth-1_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': None,
                'inference_inhibition': None,
                'conjunctive_starts': False,
                'conjunctive_successors': False,
                'conjunctive_composition': False,
                'memory_sampler': False,
                'sampler_feature_weight': 1,
                'sampler_recency_weight': 0,
                'sampler_specificity': 1
            },

        ],

        # Training targets
        'training_targets_set': np.array([

            # Block 1
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