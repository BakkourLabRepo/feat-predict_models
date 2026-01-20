import numpy as np
import pandas as pd
from os import makedirs, listdir
import pickle
from scripts.SuccessorFeatures import SuccessorFeatures
from scripts.MBRL import MBRL
from scripts.Env import Env

def check_state_match(state_1, state_2):
    """
    Returns True if the two states are identical, False otherwise.

    Arguments
    ---------
    state_1 : numpy.ndarray
        The first state to compare.
    state_2 : numpy.ndarray
        The second state to compare.

    Returns
    -------
    state_match : bool
        True if the states match, False otherwise.
    """
    state_match = np.all(state_1 == state_2)
    return state_match

def get_reward(target, successor):
    """
    Calculates reward based on the overlap of the target and successor
    features.
    
    Arguments
    ---------
    target : numpy.ndarray
        Target state.
    successor : numpy.ndarray
        Successor state.
    
    Returns
    -------
    reward : int
        Reward value
    """
    reward = np.sum(
        (target == successor) &
        (target != 0)
    )
    return reward

def train_agent(agent, env, targets, options, fixed_training=True):
    """
    Run agent on the training phase.

    Arguments
    ---------
    agent : SuccessorFeatures
        Instance of the SuccessorFeatures agent.
    env : Env
        Instance of the environment.
    targets : numpy.ndarray
        The list of training targets.
    options : numpy.ndarray
        The list of training options.
    fixed_training : bool
        If True, the agent will always compose the target's predecessor.


    Returns
    -------
    training_data : numpy.ndarray
        Simulated training data.
    """
    training_data = []
    for t in range(len(targets)):
        target = targets[t]
        options_comb = options[t]

        # Set target as task
        agent.set_task(target)
        target_comb = (target > 0).astype(int)

        # Generate feature set
        env.sample_features(
            comb = options_comb,
            terminal = False
        )

        # Set composition as predecessor of target
        if fixed_training:
            set_composition = env.get_start_state(target)
        else:
            set_composition = []

        # Get composition
        composition, p = agent.compose_from_set(
            env.a, 
            set_composition = set_composition
            )
        env.s = composition
        agent.update_memory(env.s)

        # Step environment
        while True:
            env.step()

            # Update agent memory for new state
            if not env.check_absorbing():
                agent.update_memory(env.s_new)

            # Update successor matrix
            agent.update_M(env.s, env.s_new)

            # Terminate when absorbing state is met
            if env.check_absorbing():
                break
            env.update_current_state() 

        # Get reward and whether composition was correct or not
        reward = get_reward(target, env.s_new)
        correct = int(check_state_match(target, env.s_new))

        # Store trial data
        training_data.append([
            t + 1,
            target_comb,
            options_comb,
            target,
            env.a,
            composition,
            env.s_new,
            p,
            reward,
            correct
        ])
    training_data = np.array(training_data, dtype=object)

    return training_data

def test_agent(agent, env, targets, options):
    """
    Run agent on the test phase. Trials will be all unique combinations
    of test_combs_set and test_targets.

    Arguments
    ---------
    agent : SuccessorFeatures
        Instance of the SuccessorFeatures agent.
    env : Env
        Instance of the environment.
    targets : numpy.ndarray
        The list of test targets.
    options : numpy.ndarray
        The list of test options.

    Returns
    -------
    test_data : numpy.ndarray
        Simulated test data.
    """
    test_data = []
    for t in range(len(targets)):
        target = targets[t]
        options_comb = options[t]

        # Set target as task
        agent.set_task(target)
        target_comb = (target > 0).astype(int)

        # Generate feature set
        env.sample_features(
            options_comb,
            terminal = False
        )

        # Get composition
        composition, p = agent.compose_from_set(env.a)

        # Step environment to get absorbing state, do not update agent
        env.s = composition
        while True:
            env.step()
            if env.check_absorbing():
                break
            env.update_current_state() 

        # Get reward and whether composition was correct or not
        reward = get_reward(target, env.s_new)
        correct = int(check_state_match(target, env.s_new))

        # Store trial data
        test_data.append([
            t + 1,
            target_comb,
            options_comb,
            target,
            env.a,
            composition,
            env.s_new,
            p,
            reward,
            correct
        ])
    test_data = np.array(test_data, dtype=object)

    return test_data

def simulate_agent(
        model,
        agent_config,
        env_config,
        training_trial_info,
        test_trial_info
    ):
    """
    Simulates the agent in the given environment using the provided configurations.

    Arguments
    ---------
    model : str
        The type of agent to simulate ('MBRL' or 'SuccessorFeatures').
    agent_config : dict
        Agent configuration
    env_config : dict
        Environment configuration.
    training_trial_info : dict
        Training trial targets and options
    test_trial_info : dict
        Test trial targets and options

    Returns
    -------
    training_data : numpy.ndarray
        Simulated training data.
    test_data : numpy.ndarray
        Simulated test data.
    """

    # Initialize environment 
    env = Env(**env_config)

    # Initialize agent
    if model == 'MBRL':
        agent = MBRL(env, **agent_config)
    elif model == 'SuccessorFeatures':
        agent = SuccessorFeatures(env, **agent_config)

    # Generate test target orders
    #test_targets = generate_test_targets(env)
    
    # Simulate agent
    training_data = train_agent(agent, env, **training_trial_info)
    test_data = test_agent(agent, env, **test_trial_info)

    # Get agent representations
    representations = {
        'model': model,
        'agent_info': agent_config,
        'S': agent.S,
        'F': agent.F,
        'F_raw': agent.F_raw,
        'M': agent.M,
        'bias': agent.bias,
        'bias_terminal': agent.bias_terminal,
        'recency': agent.recency,
        'frequency': agent.frequency
    }
    
    return training_data, test_data, representations

def generate_training_trial_info(
        training_targets_set,
        n_training_target_repeats
    ):
    """
    Generate training targets by repeating and permuting the target
    sets specified in training_targets_set.

    Arguments
    ---------
    training_targets_set : numpy.ndarray
        The set of target sets to be used in training.
    n_training_target_repeats : int
        The number of times each target set should be repeated.

    Returns
    -------
    trial_info : dict
        Training trial targets and options.
    """
    targets = []
    for target_set in training_targets_set:
        block_targets = np.repeat(
            target_set,
            n_training_target_repeats,
            axis=0
        )
        block_targets = np.random.permutation(block_targets)
        targets.append(block_targets)
    targets = np.array(targets)
    targets = targets.reshape(-1, len(targets[0][0]))
    trial_info = {
        'targets': targets,
        'options': (targets != 0).astype(int)
    }
    return trial_info
    
def generate_test_targets(env):
    """
    Generate test targets.

    Arguments
    ---------
    env : Env
        An instance of the agent's environment.

    Returns
    -------
    targets : numpy.ndarray
        An array of test targets with shape (n_samples, n_features).
    """
    targets = []
    for feature_comb in env.states.keys():
        targets.append(env.states[feature_comb]['terminal'])
    targets = np.array(targets).reshape(-1, env.n_feats)
    return targets

def generate_test_trial_info(
    env,
    test_combs_set
):
    """
    Generate test targets and options.

    Arguments
    ---------
    env : Env
        An instance of the agent's environment.
    test_combs_set : numpy.ndarray
        Set of test options.

    Returns
    -------
    trial_info : dict
        Test trial targets and options.
    """
    targets = generate_test_targets(env)
    options = np.repeat(test_combs_set, len(targets), axis=0)
    targets = np.tile(targets, (len(test_combs_set), 1))
    trial_info = {
        'targets': targets,
        'options': options
    }
    return trial_info

def convert_state_str(state_str):
    """
    Convert a string representation of a state to a numpy array.

    Arguments
    ---------
    state_str : str
        The string representation of the state.

    Returns
    -------
    state_arr : numpy.ndarray
        The numpy array representation of the state
    """
    state_str = state_str[1:-1].split(' ')
    state_arr = np.array(state_str, dtype=int)
    return state_arr

def transform_state_array(state_array, feature_reorder=[]):
    """
    Applies convert_state_str to a list of state strings.

    Arguments
    ---------
    state_array : list
        A list of state strings.
    feature_reorder : list
        A list of indices to reorder the features.

    Returns
    -------
    converted_state_arrat : list
        A list of converted state values.
    """
    converted_state_array = []
    for state_str in state_array:
        state = convert_state_str(state_str)
        if len(feature_reorder) > 0:
            state = state[feature_reorder]
        converted_state_array.append(state)
    converted_state_array = np.array(converted_state_array)
    return converted_state_array

def load_trial_info(trial_info_path):
    """
    Load trial information from the specified path.

    Arguments
    ---------
    trial_info_path : str
        The path to the trial information.
    
    Returns
    -------
    trial_info_set : dict
        The set of trial information.
    """

    # Check if only one set of trial orders is to be loaded
    if '.csv' in trial_info_path:
        fnames = [trial_info_path]
    else:
        fnames = listdir(trial_info_path)

    # Load trial information
    trial_info_set = {}
    for f in fnames:
        if f.endswith('.csv'):

            # Get targets and options
            trial_info = pd.read_csv(f'{trial_info_path}/{f}')
            targets = trial_info['target'].values
            options = trial_info['options_comb'].values  

            # If between-features condition, reorder target features
            feature_reorder = []  
            if 'between_cond' in trial_info.columns:
                between_cond = trial_info['between_cond'].values[0]
                if between_cond:
                    feature_reorder = [2,3,0,1]
            targets = transform_state_array(targets, feature_reorder)
            options = transform_state_array(options)
            
            # Add trial information to set
            subj = int(f.split('_')[1].split('.')[0])
            trial_info_set[subj] = {
                'targets': targets,
                'options': options
            }
    return trial_info_set

def select_trial_info(
    trial_info_set,
    match_trials_to_agents = False,
    key = None
):
    """
    Select trial information from the trial_info_set.

    Arguments
    ---------
    trial_info_set : dict or list
        The set of trial information.
    match_trials_to_agents : bool
        If True, will match trials to agent IDs
    key : str
        Specify the key to select from the trial_info_set.
    
    Returns
    -------
    trial_info : dict or bool
        The selected trial information. If match_trials_to_agents = True
        but there is not trial information for the agent ID, return
        False. This will skip this simulation.
    """
    if not match_trials_to_agents:
        key = np.random.choice(list(trial_info_set.keys()))
    elif key not in trial_info_set.keys():
        return False
    trial_info = trial_info_set[key]
    return trial_info


def load_configs(agent_configs_path):
    """
    Load agent configurations from the specified path.

    Arguments
    ---------
    agent_configs_path : str
        The path to the agent configurations.
    
    Returns
    -------
    agent_configs : list
        A list of agent configurations.
    """
    agent_configs = []
    for f in listdir(agent_configs_path):
        if f.endswith('.pkl'):
            with open(f'{agent_configs_path}/{f}', 'rb') as file:
                agent_configs.append(pickle.load(file))
    return agent_configs

def generate_agent_configs(n_agents, model_configs):
    """
    Generate agent configurations based on the given number of agents
    and model configurations.

    Arguments
    ---------
    n_agents : int
        The number of agents to generate per model.
    model_configs : list
        A list of basic model configurations.

    Returns
    -------
    agent_configs : list
        A list of agent configurations with sampled parameters.
    """
    agent_configs = []
    subj = 0
    for model_config in model_configs:
        model, model_config = model_config
        for _ in range(n_agents):
            subj += 1
            agent_config = model_config.copy()
            agent_config['id'] = subj

            # Sample parameters if they are not set
            for key in agent_config.keys():
                if agent_config[key] is None:
                    if key == 'beta':
                        agent_config[key] = 1/np.random.uniform(0, 1) - 1
                    elif key in ['sampler_specificity', 'inference_inhibition']:
                        agent_config[key] = 1/np.random.uniform(0, 1)
                    else:
                        agent_config[key] = np.random.uniform(0, 1)

            # Set parameter to match value of specified parameter
            for key in agent_config.keys():
                if agent_config[key] in agent_config.keys():
                    agent_config[key] = agent_config[agent_config[key]]
                    
            agent_configs.append((model, agent_config))

    return agent_configs    

def run_experiment(
        n_agents = None,
        env_config = None,
        training_targets_set = None,
        n_training_target_repeats = None,
        test_combs_set = None,
        fixed_training = False,
        agent_configs_path = False,
        training_trial_info_path = False,
        test_trial_info_path = False,
        match_trials_to_agents = False,
        model_configs = None,
        output_path = False,
        seed = None
    ):
    """
    Simulate agents.

    Arguments
    ---------
    n_agents : int
        The number of agents to generate per model.
    env_config : dict
        Environment configuration.
    training_targets_set : numpy.ndarray
        The set of target sets to be used in training.
    n_training_target_repeats : int
        The number of times each target set should be repeated.
    test_combs_set : numpy.ndarray
        Set of test options.
    fixed_training : bool
        If True, the agent will always compose the target's predecessor.
    agent_configs_path : str
        Path to load fit agent configurations. If left as False,
        will generate new configurations.
    training_trial_info_path : str
        Path to load training trial information. If left as False,
        will generate training trials.
    test_trial_info_path : str
        Path to load test trial information. If left as False,
        will generate test trials.
    match_trials_to_agents : bool
        If True, will match loaded trials to fit agent IDs.
    model_configs : list
        A list of basic model configurations.
    output_path : str
        The path to save the data. If left as False, will not save data.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    None
    """

    np.random.seed(seed)

    # Load all agent configurations
    if agent_configs_path:
        agent_configs = []
        for subj in listdir(agent_configs_path):
            if subj == '.DS_Store': continue
            agent_configs.extend(load_configs(f'{agent_configs_path}/{subj}'))

    # Generate all agent configurations
    else:
        agent_configs = generate_agent_configs(n_agents, model_configs)

    # Load all trial information
    if training_trial_info_path:
        training_trial_info_set = load_trial_info(training_trial_info_path)
    if test_trial_info_path:
        test_trial_info_set = load_trial_info(test_trial_info_path)

    # Simulate all agents
    for model, agent_config in agent_configs:
        subj = agent_config['id']
        model_label = agent_config['model_label']

        # Create the directories to save data to
        if output_path:
            makedirs(
                f'{output_path}/{model}/{model_label}/training',
                exist_ok = True
                )
            makedirs(
                f'{output_path}/{model}/{model_label}/test',
                exist_ok = True
                )
            makedirs(
                f'{output_path}/{model}/{model_label}/representations',
                exist_ok = True
                )
            
        # Load training trials
        if training_trial_info_path:
            training_trial_info = select_trial_info(
                training_trial_info_set,
                match_trials_to_agents = match_trials_to_agents,
                key = subj
            )
            if not training_trial_info:
                continue
                
        # Generate training trials
        else:
            training_trial_info = generate_training_trial_info(
                training_targets_set,
                n_training_target_repeats
            )

        # Load test trials
        if test_trial_info_path:
            test_trial_info = select_trial_info(
                test_trial_info_set,
                match_trials_to_agents = match_trials_to_agents,
                key = subj
            )
            if not test_trial_info:
                continue
        
        # Generate test trials
        else:
            test_trial_info = generate_test_trial_info(
                Env(**env_config),
                test_combs_set
            )

        # Set whether to fix training choices or not
        training_trial_info['fixed_training'] = fixed_training
        
        # Simulate agent
        print(f'Simulating - Agent: {subj}, Model: {model} {model_label}')
        training_data, test_data, representations = simulate_agent(
            model,
            agent_config,
            env_config,
            training_trial_info,
            test_trial_info
        )

        # Convert data to dataframe
        data_colnames = [
            'trial',
            'target_comb',
            'options_comb',
            'target',
            'options',
            'composition',
            'successor',
            'p',
            'reward',
            'correct'
        ]
        training_df = pd.DataFrame(training_data, columns=data_colnames)
        test_df = pd.DataFrame(test_data, columns=data_colnames)

        # Add agent information to data
        training_df.insert(0, 'model', model)
        test_df.insert(0, 'model', model)
        for key in [*agent_config][::-1]:
            training_df.insert(0, key, agent_config[key])
            test_df.insert(0, key, agent_config[key])

        # Save data 
        if output_path:
            model_path = f'{output_path}/{model}/{model_label}'
            training_df.to_csv(
                f'{model_path}/training/training_{subj}.csv',
                index = False
            )
            test_df.to_csv(
                f'{model_path}/test/test_{subj}.csv',
                index = False
            )
            with open(
                f'{model_path}/representations/representations_{subj}.pkl',
                'wb'
            ) as file:
                pickle.dump(representations, file)