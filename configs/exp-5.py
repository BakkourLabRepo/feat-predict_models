import numpy as np

PROJECT_PATH = '/Users/euanprentis/Documents/feat_predict_simulations_2'

AGENT_CONFIGS_PATH = False

experiment_config = {
    

    'depth-4_dim-2': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-5/data/depth-4_dim-2',

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
                'SuccessorFeatures',
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
                'SuccessorFeatures',
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
                [1 ,1 ,0 ,0 ],
                [1 ,10,0 ,0 ],
                [1 ,10,0 ,0 ],
                [10,1 ,0 ,0 ],
                [10,1 ,0 ,0 ],
                [10,10,0 ,0 ],
                [2 ,2 ,0 ,0 ],
                [2 ,9 ,0 ,0 ],
                [2 ,9 ,0 ,0 ],
                [9 ,2 ,0 ,0 ],
                [9 ,2 ,0 ,0 ],
                [9 ,9 ,0 ,0 ],
                [3 ,3 ,0 ,0 ],
                [3 ,8 ,0 ,0 ],
                [3 ,8 ,0 ,0 ],
                [8 ,3 ,0 ,0 ],
                [8 ,3 ,0 ,0 ],
                [8 ,8 ,0 ,0 ],
                [4 ,4 ,0 ,0 ],
                [4 ,7 ,0 ,0 ],
                [4 ,7 ,0 ,0 ],
                [7 ,4 ,0 ,0 ],
                [7 ,4 ,0 ,0 ],
                [7 ,7 ,0 ,0 ],
                [0 ,0 ,1 ,1 ],
                [0 ,0 ,1 ,10],
                [0 ,0 ,1 ,10],
                [0 ,0 ,10,1 ],
                [0 ,0 ,10,1 ],
                [0 ,0 ,10,10],
                [0 ,0 ,2 ,2 ],
                [0 ,0 ,2 ,9 ],
                [0 ,0 ,2 ,9 ],
                [0 ,0 ,9 ,2 ],
                [0 ,0 ,9 ,2 ],
                [0 ,0 ,9 ,9 ],
                [0 ,0 ,3 ,3 ],
                [0 ,0 ,3 ,8 ],
                [0 ,0 ,3 ,8 ],
                [0 ,0 ,8 ,3 ],
                [0 ,0 ,8 ,3 ],
                [0 ,0 ,8 ,8 ],
                [0 ,0 ,4 ,4 ],
                [0 ,0 ,4 ,7 ],
                [0 ,0 ,4 ,7 ],
                [0 ,0 ,7 ,4 ],
                [0 ,0 ,7 ,4 ],
                [0 ,0 ,7 ,7 ]
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
            'max_steps': 1,
            'n_feats': 4,
            'n_fixed': 0,
            'n_per': 2,
            'start_insts': np.array([
                [5, 6],
                [4, 5],
                [3, 4],
                [2, 1]
                ]),
            'r': np.array([[-1,0,0,0,0,0,0,0,0,1]]),
            'continuous_features': False
        },

    },

}