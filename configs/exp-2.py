import numpy as np

PROJECT_PATH = ''

AGENT_CONFIGS_PATH = ''

experiment_config = {

   
    'depth-1_dim-2_ntrain-1080': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-1_dim-2_ntrain-1080',

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
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-1_dim-2_ntrain-72': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-1_dim-2_ntrain-72',

        # Random seed for reproducibility
        'seed': 243423,

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
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-1_dim-4_ntrain-1080': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-1_dim-4_ntrain-1080',

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
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,1,1,1,0,0,0,0],
                [1,1,4,4,0,0,0,0],
                [1,1,4,4,0,0,0,0],
                [4,4,1,1,0,0,0,0],
                [4,4,1,1,0,0,0,0],
                [4,4,4,4,0,0,0,0],
                [0,0,0,0,1,1,1,1],
                [0,0,0,0,1,1,4,4],
                [0,0,0,0,1,1,4,4],
                [0,0,0,0,4,4,1,1],
                [0,0,0,0,4,4,1,1],
                [0,0,0,0,4,4,4,4]
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
                [1,0,0,0],
                [1,0,0,0],
                [0,0,0,1],
                [0,0,0,1]
            ]),
            'n_feats': 8,
            'n_fixed': 0,
            'n_per': 4,
            'start_insts': np.array([2, 3]),
            'r': np.array([[-1,0,0,1]]),
            'continuous_features': False
        },

    },

    'depth-1_dim-4_ntrain-72': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-1_dim-4_ntrain-72',

        # Random seed for reproducibility
        'seed': 243423,

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
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

        # Training targets
        'training_targets_set': np.array([

            [
                [1,1,1,1,0,0,0,0],
                [1,1,4,4,0,0,0,0],
                [1,1,4,4,0,0,0,0],
                [4,4,1,1,0,0,0,0],
                [4,4,1,1,0,0,0,0],
                [4,4,4,4,0,0,0,0],
                [0,0,0,0,1,1,1,1],
                [0,0,0,0,1,1,4,4],
                [0,0,0,0,1,1,4,4],
                [0,0,0,0,4,4,1,1],
                [0,0,0,0,4,4,1,1],
                [0,0,0,0,4,4,4,4]
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
                [1,0,0,0],
                [1,0,0,0],
                [0,0,0,1],
                [0,0,0,1]
            ]),
            'n_feats': 8,
            'n_fixed': 0,
            'n_per': 4,
            'start_insts': np.array([2, 3]),
            'r': np.array([[-1,0,0,1]]),
            'continuous_features': False
        },

    },


    'depth-2_dim-2_ntrain-1080': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-2_dim-2_ntrain-1080',

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
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-2_dim-2_ntrain-72': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-2_dim-2_ntrain-72',

        # Random seed for reproducibility
        'seed': 243423,

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
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-2_dim-4_ntrain-1080': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-2_dim-4_ntrain-1080',

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
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-2_dim-4_ntrain-72': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-2_dim-4_ntrain-72',

        # Random seed for reproducibility
        'seed': 243423,

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
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-3_dim-2_ntrain-1080': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-3_dim-2_ntrain-1080',

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
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-3_dim-2_ntrain-72': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-3_dim-2_ntrain-72',

        # Random seed for reproducibility
        'seed': 243423,

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
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-3_dim-4_ntrain-1080': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-3_dim-4_ntrain-1080',

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
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-3_dim-4_ntrain-72': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-3_dim-4_ntrain-72',

        # Random seed for reproducibility
        'seed': 243423,

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
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-4_dim-2_ntrain-1080': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-4_dim-2_ntrain-1080',

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
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-4_dim-2_ntrain-72': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-4_dim-2_ntrain-72',

        # Random seed for reproducibility
        'seed': 243423,

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
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-4_dim-4_ntrain-1080': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-4_dim-4_ntrain-1080',

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
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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


    'depth-4_dim-4_ntrain-72': {

        # Output directory
        'output_path': f'{PROJECT_PATH}/exp-2/data/depth-4_dim-4_ntrain-72',

        # Random seed for reproducibility
        'seed': 243423,

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
        'n_agents': None,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [],

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