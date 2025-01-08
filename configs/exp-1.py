import numpy as np

experiment_config = {
    
    'model-ff_bias-none_sdim-1_depth-1_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 270,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-1_depth-1_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,0,0,0],
                [4,0,0,0],
                [0,1,0,0],
                [0,4,0,0],
                [0,0,1,0],
                [0,0,4,0],
                [0,0,0,1],
                [0,0,0,4]
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
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




    'model-ff_bias-none_sdim-1_depth-2_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 270,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-1_depth-2_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,0,0,0],
                [6,0,0,0],
                [0,1,0,0],
                [0,6,0,0],
                [0,0,1,0],
                [0,0,6,0],
                [0,0,0,1],
                [0,0,0,6]
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
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



    'model-ff_bias-none_sdim-1_depth-3_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 270,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-1_depth-3_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,0,0,0],
                [8,0,0,0],
                [0,1,0,0],
                [0,8,0,0],
                [0,0,1,0],
                [0,0,8,0],
                [0,0,0,1],
                [0,0,0,8]
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
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





    'model-ff_bias-none_sdim-1_depth-4_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 270,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-1_depth-4_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,0,0,0],
                [10,0,0,0],
                [0,1,0,0],
                [0,10,0,0],
                [0,0,1,0],
                [0,0,10,0],
                [0,0,0,1],
                [0,0,0,10]
            ]

        ]),

        # Test feature combinations in the composition set
        'test_combs_set': np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
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




    'model-ff_bias-none_sdim-2_depth-1_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-1_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [4,1,0,0],
                [4,4,0,0],
                [0,0,1,1],
                [0,0,1,4],
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


    'model-ff_bias-true_sdim-2_depth-1_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-1_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [4,1,0,0],
                [4,4,0,0],
                [0,0,1,1],
                [0,0,1,4],
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



    'model-ff_bias-incidental_sdim-2_depth-1_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-1_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [4,1,0,0],
                [4,4,0,0],
                [0,0,1,1],
                [0,0,1,4],
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



    'model-ff_bias-none_sdim-2_depth-2_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-2_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,6,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [0,0,1,1],
                [0,0,1,6],
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


    'model-ff_bias-true_sdim-2_depth-2_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-2_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,6,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [0,0,1,1],
                [0,0,1,6],
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



    'model-ff_bias-incidental_sdim-2_depth-2_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-2_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,6,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [0,0,1,1],
                [0,0,1,6],
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



    'model-ff_bias-none_sdim-2_depth-3_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-3_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,8,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [0,0,1,1],
                [0,0,1,8],
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



    'model-ff_bias-true_sdim-2_depth-3_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-3_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,8,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [0,0,1,1],
                [0,0,1,8],
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



    'model-ff_bias-incidental_sdim-2_depth-3_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-3_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,8,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [0,0,1,1],
                [0,0,1,8],
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



    'model-ff_bias-none_sdim-2_depth-4_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-4_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,10,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [0,0,1,1],
                [0,0,1,10],
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



    'model-ff_bias-true_sdim-2_depth-4_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-4_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,10,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [0,0,1,1],
                [0,0,1,10],
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



    'model-ff_bias-incidental_sdim-2_depth-4_freq-uniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 135,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-4_freq-uniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,10,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [0,0,1,1],
                [0,0,1,10],
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










    'model-ff_bias-none_sdim-2_depth-1_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

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
                'segmentation': 0,
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


    'model-ff_bias-true_sdim-2_depth-1_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-1_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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



    'model-ff_bias-incidental_sdim-2_depth-1_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-1_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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



    'model-ff_bias-none_sdim-2_depth-2_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-2_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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


    'model-ff_bias-true_sdim-2_depth-2_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-2_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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



    'model-ff_bias-incidental_sdim-2_depth-2_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-2_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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



    'model-ff_bias-none_sdim-2_depth-3_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-3_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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



    'model-ff_bias-true_sdim-2_depth-3_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-3_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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



    'model-ff_bias-incidental_sdim-2_depth-3_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-3_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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



    'model-ff_bias-none_sdim-2_depth-4_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-4_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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



    'model-ff_bias-true_sdim-2_depth-4_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-4_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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



    'model-ff_bias-incidental_sdim-2_depth-4_freq-nonuniform_pairs-fact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

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
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-4_freq-nonuniform_pairs-fact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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

    



















    'model-ff_bias-none_sdim-2_depth-1_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-1_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [4,1,0,0],
                [4,4,0,0],
                [1,0,1,0],
                [1,0,4,0],
                [4,0,1,0],
                [4,0,4,0],
                [1,0,0,1],
                [1,0,0,4],
                [4,0,0,1],
                [4,0,0,4],
                [0,1,1,0],
                [0,1,4,0],
                [0,4,1,0],
                [0,4,4,0],
                [0,1,0,1],
                [0,1,0,4],
                [0,4,0,1],
                [0,4,0,4],
                [0,0,1,1],
                [0,0,1,4],
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


    'model-ff_bias-true_sdim-2_depth-1_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-1_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [4,1,0,0],
                [4,4,0,0],
                [1,0,1,0],
                [1,0,4,0],
                [4,0,1,0],
                [4,0,4,0],
                [1,0,0,1],
                [1,0,0,4],
                [4,0,0,1],
                [4,0,0,4],
                [0,1,1,0],
                [0,1,4,0],
                [0,4,1,0],
                [0,4,4,0],
                [0,1,0,1],
                [0,1,0,4],
                [0,4,0,1],
                [0,4,0,4],
                [0,0,1,1],
                [0,0,1,4],
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



    'model-ff_bias-incidental_sdim-2_depth-1_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-1_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [4,1,0,0],
                [4,4,0,0],
                [1,0,1,0],
                [1,0,4,0],
                [4,0,1,0],
                [4,0,4,0],
                [1,0,0,1],
                [1,0,0,4],
                [4,0,0,1],
                [4,0,0,4],
                [0,1,1,0],
                [0,1,4,0],
                [0,4,1,0],
                [0,4,4,0],
                [0,1,0,1],
                [0,1,0,4],
                [0,4,0,1],
                [0,4,0,4],
                [0,0,1,1],
                [0,0,1,4],
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



    'model-ff_bias-none_sdim-2_depth-2_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-2_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,6,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [1,0,1,0],
                [1,0,6,0],
                [6,0,1,0],
                [6,0,6,0],
                [1,0,0,1],
                [1,0,0,6],
                [6,0,0,1],
                [6,0,0,6],
                [0,1,1,0],
                [0,1,6,0],
                [0,6,1,0],
                [0,6,6,0],
                [0,1,0,1],
                [0,1,0,6],
                [0,6,0,1],
                [0,6,0,6],
                [0,0,1,1],
                [0,0,1,6],
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


    'model-ff_bias-true_sdim-2_depth-2_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-2_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,6,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [1,0,1,0],
                [1,0,6,0],
                [6,0,1,0],
                [6,0,6,0],
                [1,0,0,1],
                [1,0,0,6],
                [6,0,0,1],
                [6,0,0,6],
                [0,1,1,0],
                [0,1,6,0],
                [0,6,1,0],
                [0,6,6,0],
                [0,1,0,1],
                [0,1,0,6],
                [0,6,0,1],
                [0,6,0,6],
                [0,0,1,1],
                [0,0,1,6],
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



    'model-ff_bias-incidental_sdim-2_depth-2_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-2_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,6,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [1,0,1,0],
                [1,0,6,0],
                [6,0,1,0],
                [6,0,6,0],
                [1,0,0,1],
                [1,0,0,6],
                [6,0,0,1],
                [6,0,0,6],
                [0,1,1,0],
                [0,1,6,0],
                [0,6,1,0],
                [0,6,6,0],
                [0,1,0,1],
                [0,1,0,6],
                [0,6,0,1],
                [0,6,0,6],
                [0,0,1,1],
                [0,0,1,6],
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



    'model-ff_bias-none_sdim-2_depth-3_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-3_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,8,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [1,0,1,0],
                [1,0,8,0],
                [8,0,1,0],
                [8,0,8,0],
                [1,0,0,1],
                [1,0,0,8],
                [8,0,0,1],
                [8,0,0,8],
                [0,1,1,0],
                [0,1,8,0],
                [0,8,1,0],
                [0,8,8,0],
                [0,1,0,1],
                [0,1,0,8],
                [0,8,0,1],
                [0,8,0,8],
                [0,0,1,1],
                [0,0,1,8],
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



    'model-ff_bias-true_sdim-2_depth-3_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-3_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,8,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [1,0,1,0],
                [1,0,8,0],
                [8,0,1,0],
                [8,0,8,0],
                [1,0,0,1],
                [1,0,0,8],
                [8,0,0,1],
                [8,0,0,8],
                [0,1,1,0],
                [0,1,8,0],
                [0,8,1,0],
                [0,8,8,0],
                [0,1,0,1],
                [0,1,0,8],
                [0,8,0,1],
                [0,8,0,8],
                [0,0,1,1],
                [0,0,1,8],
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



    'model-ff_bias-incidental_sdim-2_depth-3_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-3_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,8,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [1,0,1,0],
                [1,0,8,0],
                [8,0,1,0],
                [8,0,8,0],
                [1,0,0,1],
                [1,0,0,8],
                [8,0,0,1],
                [8,0,0,8],
                [0,1,1,0],
                [0,1,8,0],
                [0,8,1,0],
                [0,8,8,0],
                [0,1,0,1],
                [0,1,0,8],
                [0,8,0,1],
                [0,8,0,8],
                [0,0,1,1],
                [0,0,1,8],
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



    'model-ff_bias-none_sdim-2_depth-4_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-4_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,10,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [1,0,1,0],
                [1,0,10,0],
                [10,0,1,0],
                [10,0,10,0],
                [1,0,0,1],
                [1,0,0,10],
                [10,0,0,1],
                [10,0,0,10],
                [0,1,1,0],
                [0,1,10,0],
                [0,10,1,0],
                [0,10,10,0],
                [0,1,0,1],
                [0,1,0,10],
                [0,10,0,1],
                [0,10,0,10],
                [0,0,1,1],
                [0,0,1,10],
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



    'model-ff_bias-true_sdim-2_depth-4_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-4_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,10,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [1,0,1,0],
                [1,0,10,0],
                [10,0,1,0],
                [10,0,10,0],
                [1,0,0,1],
                [1,0,0,10],
                [10,0,0,1],
                [10,0,0,10],
                [0,1,1,0],
                [0,1,10,0],
                [0,10,1,0],
                [0,10,10,0],
                [0,1,0,1],
                [0,1,0,10],
                [0,10,0,1],
                [0,10,0,10],
                [0,0,1,1],
                [0,0,1,10],
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



    'model-ff_bias-incidental_sdim-2_depth-4_freq-uniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 45,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-4_freq-uniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,10,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [1,0,1,0],
                [1,0,10,0],
                [10,0,1,0],
                [10,0,10,0],
                [1,0,0,1],
                [1,0,0,10],
                [10,0,0,1],
                [10,0,0,10],
                [0,1,1,0],
                [0,1,10,0],
                [0,10,1,0],
                [0,10,10,0],
                [0,1,0,1],
                [0,1,0,10],
                [0,10,0,1],
                [0,10,0,10],
                [0,0,1,1],
                [0,0,1,10],
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


















    'model-ff_bias-none_sdim-2_depth-1_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-1_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,0,1,0],
                [1,0,4,0],
                [1,0,4,0],
                [4,0,1,0],
                [4,0,1,0],
                [4,0,4,0],
                [1,0,0,1],
                [1,0,0,4],
                [1,0,0,4],
                [4,0,0,1],
                [4,0,0,1],
                [4,0,0,4],
                [0,1,1,0],
                [0,1,4,0],
                [0,1,4,0],
                [0,4,1,0],
                [0,4,1,0],
                [0,4,4,0],
                [0,1,0,1],
                [0,1,0,4],
                [0,1,0,4],
                [0,4,0,1],
                [0,4,0,1],
                [0,4,0,4],
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


    'model-ff_bias-true_sdim-2_depth-1_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-1_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,0,1,0],
                [1,0,4,0],
                [1,0,4,0],
                [4,0,1,0],
                [4,0,1,0],
                [4,0,4,0],
                [1,0,0,1],
                [1,0,0,4],
                [1,0,0,4],
                [4,0,0,1],
                [4,0,0,1],
                [4,0,0,4],
                [0,1,1,0],
                [0,1,4,0],
                [0,1,4,0],
                [0,4,1,0],
                [0,4,1,0],
                [0,4,4,0],
                [0,1,0,1],
                [0,1,0,4],
                [0,1,0,4],
                [0,4,0,1],
                [0,4,0,1],
                [0,4,0,4],
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



    'model-ff_bias-incidental_sdim-2_depth-1_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-1_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,0,1,0],
                [1,0,4,0],
                [1,0,4,0],
                [4,0,1,0],
                [4,0,1,0],
                [4,0,4,0],
                [1,0,0,1],
                [1,0,0,4],
                [1,0,0,4],
                [4,0,0,1],
                [4,0,0,1],
                [4,0,0,4],
                [0,1,1,0],
                [0,1,4,0],
                [0,1,4,0],
                [0,4,1,0],
                [0,4,1,0],
                [0,4,4,0],
                [0,1,0,1],
                [0,1,0,4],
                [0,1,0,4],
                [0,4,0,1],
                [0,4,0,1],
                [0,4,0,4],
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



    'model-ff_bias-none_sdim-2_depth-2_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-2_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,6,0,0],
                [1,6,0,0],
                [6,1,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [1,0,1,0],
                [1,0,6,0],
                [1,0,6,0],
                [6,0,1,0],
                [6,0,1,0],
                [6,0,6,0],
                [1,0,0,1],
                [1,0,0,6],
                [1,0,0,6],
                [6,0,0,1],
                [6,0,0,1],
                [6,0,0,6],
                [0,1,1,0],
                [0,1,6,0],
                [0,1,6,0],
                [0,6,1,0],
                [0,6,1,0],
                [0,6,6,0],
                [0,1,0,1],
                [0,1,0,6],
                [0,1,0,6],
                [0,6,0,1],
                [0,6,0,1],
                [0,6,0,6],
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


    'model-ff_bias-true_sdim-2_depth-2_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-2_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,6,0,0],
                [1,6,0,0],
                [6,1,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [1,0,1,0],
                [1,0,6,0],
                [1,0,6,0],
                [6,0,1,0],
                [6,0,1,0],
                [6,0,6,0],
                [1,0,0,1],
                [1,0,0,6],
                [1,0,0,6],
                [6,0,0,1],
                [6,0,0,1],
                [6,0,0,6],
                [0,1,1,0],
                [0,1,6,0],
                [0,1,6,0],
                [0,6,1,0],
                [0,6,1,0],
                [0,6,6,0],
                [0,1,0,1],
                [0,1,0,6],
                [0,1,0,6],
                [0,6,0,1],
                [0,6,0,1],
                [0,6,0,6],
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



    'model-ff_bias-incidental_sdim-2_depth-2_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-2_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,6,0,0],
                [1,6,0,0],
                [6,1,0,0],
                [6,1,0,0],
                [6,6,0,0],
                [1,0,1,0],
                [1,0,6,0],
                [1,0,6,0],
                [6,0,1,0],
                [6,0,1,0],
                [6,0,6,0],
                [1,0,0,1],
                [1,0,0,6],
                [1,0,0,6],
                [6,0,0,1],
                [6,0,0,1],
                [6,0,0,6],
                [0,1,1,0],
                [0,1,6,0],
                [0,1,6,0],
                [0,6,1,0],
                [0,6,1,0],
                [0,6,6,0],
                [0,1,0,1],
                [0,1,0,6],
                [0,1,0,6],
                [0,6,0,1],
                [0,6,0,1],
                [0,6,0,6],
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



    'model-ff_bias-none_sdim-2_depth-3_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-3_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,8,0,0],
                [1,8,0,0],
                [8,1,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [1,0,1,0],
                [1,0,8,0],
                [1,0,8,0],
                [8,0,1,0],
                [8,0,1,0],
                [8,0,8,0],
                [1,0,0,1],
                [1,0,0,8],
                [1,0,0,8],
                [8,0,0,1],
                [8,0,0,1],
                [8,0,0,8],
                [0,1,1,0],
                [0,1,8,0],
                [0,1,8,0],
                [0,8,1,0],
                [0,8,1,0],
                [0,8,8,0],
                [0,1,0,1],
                [0,1,0,8],
                [0,1,0,8],
                [0,8,0,1],
                [0,8,0,1],
                [0,8,0,8],
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



    'model-ff_bias-true_sdim-2_depth-3_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-3_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,8,0,0],
                [1,8,0,0],
                [8,1,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [1,0,1,0],
                [1,0,8,0],
                [1,0,8,0],
                [8,0,1,0],
                [8,0,1,0],
                [8,0,8,0],
                [1,0,0,1],
                [1,0,0,8],
                [1,0,0,8],
                [8,0,0,1],
                [8,0,0,1],
                [8,0,0,8],
                [0,1,1,0],
                [0,1,8,0],
                [0,1,8,0],
                [0,8,1,0],
                [0,8,1,0],
                [0,8,8,0],
                [0,1,0,1],
                [0,1,0,8],
                [0,1,0,8],
                [0,8,0,1],
                [0,8,0,1],
                [0,8,0,8],
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



    'model-ff_bias-incidental_sdim-2_depth-3_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-3_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,8,0,0],
                [1,8,0,0],
                [8,1,0,0],
                [8,1,0,0],
                [8,8,0,0],
                [1,0,1,0],
                [1,0,8,0],
                [1,0,8,0],
                [8,0,1,0],
                [8,0,1,0],
                [8,0,8,0],
                [1,0,0,1],
                [1,0,0,8],
                [1,0,0,8],
                [8,0,0,1],
                [8,0,0,1],
                [8,0,0,8],
                [0,1,1,0],
                [0,1,8,0],
                [0,1,8,0],
                [0,8,1,0],
                [0,8,1,0],
                [0,8,8,0],
                [0,1,0,1],
                [0,1,0,8],
                [0,1,0,8],
                [0,8,0,1],
                [0,8,0,1],
                [0,8,0,8],
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



    'model-ff_bias-none_sdim-2_depth-4_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-none_sdim-2_depth-4_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 0,
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
                [1,10,0,0],
                [1,10,0,0],
                [10,1,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [1,0,1,0],
                [1,0,10,0],
                [1,0,10,0],
                [10,0,1,0],
                [10,0,1,0],
                [10,0,10,0],
                [1,0,0,1],
                [1,0,0,10],
                [1,0,0,10],
                [10,0,0,1],
                [10,0,0,1],
                [10,0,0,10],
                [0,1,1,0],
                [0,1,10,0],
                [0,1,10,0],
                [0,10,1,0],
                [0,10,1,0],
                [0,10,10,0],
                [0,1,0,1],
                [0,1,0,10],
                [0,1,0,10],
                [0,10,0,1],
                [0,10,0,1],
                [0,10,0,10],
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



    'model-ff_bias-true_sdim-2_depth-4_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-true_sdim-2_depth-4_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': 1,
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
                [1,10,0,0],
                [1,10,0,0],
                [10,1,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [1,0,1,0],
                [1,0,10,0],
                [1,0,10,0],
                [10,0,1,0],
                [10,0,1,0],
                [10,0,10,0],
                [1,0,0,1],
                [1,0,0,10],
                [1,0,0,10],
                [10,0,0,1],
                [10,0,0,1],
                [10,0,0,10],
                [0,1,1,0],
                [0,1,10,0],
                [0,1,10,0],
                [0,10,1,0],
                [0,10,1,0],
                [0,10,10,0],
                [0,1,0,1],
                [0,1,0,10],
                [0,1,0,10],
                [0,10,0,1],
                [0,10,0,1],
                [0,10,0,10],
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



    'model-ff_bias-incidental_sdim-2_depth-4_freq-nonuniform_pairs-nonfact': {

        # Output directory
        'output_path': '/Users/euanprentis/Documents/feat_predict_simulations/exp-1/data',

        # Random seed for reproducibility
        'seed': 9823,

        # Number of training trials
        'n_training_target_repeats': 30,

        # Simulate based on existing agent configurations
        'agent_configs_path': False,

        # Load existing trial information
        'training_trial_info_path': False,
        'test_trial_info_path': False,
        'match_trials_to_agents': False,

        # Number of agents per basic agent config
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'n_agents': 250,

        # Force training compositions to be of the target's predecessor 
        'fixed_training': False,

        # Configurations for models to simulate
        # Only need to set if LOAD_AGENT_CONFIGS = False
        'model_configs': [

            # Feature -> Feature model
            {
                'id': None,
                'model_label': 'model-ff_bias-incidental_sdim-2_depth-4_freq-nonuniform_pairs-nonfact',
                'alpha': None, 
                'alpha_decay': 0, 
                'beta': None,
                'beta_test': 'beta',
                'gamma': 1,
                'segmentation': -1,
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
                [1,10,0,0],
                [1,10,0,0],
                [10,1,0,0],
                [10,1,0,0],
                [10,10,0,0],
                [1,0,1,0],
                [1,0,10,0],
                [1,0,10,0],
                [10,0,1,0],
                [10,0,1,0],
                [10,0,10,0],
                [1,0,0,1],
                [1,0,0,10],
                [1,0,0,10],
                [10,0,0,1],
                [10,0,0,1],
                [10,0,0,10],
                [0,1,1,0],
                [0,1,10,0],
                [0,1,10,0],
                [0,10,1,0],
                [0,10,1,0],
                [0,10,10,0],
                [0,1,0,1],
                [0,1,0,10],
                [0,1,0,10],
                [0,10,0,1],
                [0,10,0,1],
                [0,10,0,10],
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




}