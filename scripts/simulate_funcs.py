import numpy as np
import pandas as pd
from os import makedirs, listdir
import pickle
from scripts.Successor_Features import Successor_Features
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

def train_agent(agent, env, training_targets):
    """
    Run agent on the training phase.

    Arguments
    ---------
    agent : Successor_Features
        Instance of the Successor_Features agent.
    env : Env
        Instance of the environment.
    training_targets : numpy.ndarray
        The list of training targets.

    Returns
    -------
    training_data : numpy.ndarray
        Simulated training data.
    """
    training_data = []
    for t, target in enumerate(training_targets):

        # Set target as task
        agent.set_task(target)
        target_comb = (target > 0).astype(int)

        # Generate feature set
        options_comb = target_comb
        env.sample_features(
            comb = options_comb,
            terminal = False
        )

        # Get composition
        composition, p = agent.compose_from_set(env.a)
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

def test_agent(agent, env, test_combs_set, test_targets):
    """
    Run agent on the test phase. Trials will be all unique combinations
    of test_combs_set and test_targets.

    Arguments
    ---------
    agent : Successor_Features
        Instance of the Successor_Features agent.
    env : Env
        Instance of the environment.
    test_combs_set : numpy.ndarray
        Set of options.
    test_targets : numpy.ndarray
        Set of targets.

    Returns
    -------
    test_data : numpy.ndarray
        Simulated test data.
    """
    test_data = []
    t = 0
    for target in test_targets:

        # Set target as task
        agent.set_task(target)
        target_comb = (target > 0).astype(int)

        # Test target inference for every test feature set
        for options_comb in test_combs_set:
            t += 1

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

def generate_training_targets(training_targets_set, n_training_target_repeats):
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
    targets : numpy.ndarray
        An array of training targets with shape (n_samples, n_features).
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
    return targets
    
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

def simulate_agent(
        agent_config,
        env_config,
        training_targets_set,
        n_training_target_repeats,
        test_combs_set
    ):
    """
    Simulates the agent in the given environment using the provided configurations.

    Arguments
    ---------
    agent_config : dict
        Agent configuration
    env_config : dict
        Environment configuration.
    training_targets_set : numpy.ndarray
        The set of target sets to be used in training.
    n_training_target_repeats : int
        The number of times each target set should be repeated.
    test_combs_set : list
        Set of test options

    Returns
    -------
    training_data : numpy.ndarray
        Simulated training data.
    test_data : numpy.ndarray
        Simulated test data.
    """

    # Initialize environment and agent
    env = Env(**env_config)
    agent = Successor_Features(env, **agent_config)

    # Generate training and test target orders
    training_targets = generate_training_targets(
        training_targets_set,
        n_training_target_repeats
    )
    test_targets = generate_test_targets(env)
    
    # Simulate agent
    training_data = train_agent(agent, env, training_targets)
    agent.beta = agent.beta_test
    test_data = test_agent(agent, env, test_combs_set, test_targets)

    # Get agent representations
    representations = {
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
        for _ in range(n_agents):
            subj += 1
            agent_config = model_config.copy()
            agent_config['id'] = subj

            # Sample parameters if they are not set
            for key in agent_config.keys():
                if agent_config[key] is None:
                    if key == 'beta' or key == 'beta_test':
                        agent_config[key] = 1/np.random.uniform(0, 1) - 1
                    elif key == 'sampler_specificity':
                        agent_config[key] = 1/np.random.uniform(0, 1)
                    else:
                        agent_config[key] = np.random.uniform(0, 1)

            # Set parameter to match value of specified parameter
            for key in agent_config.keys():
                if agent_config[key] in agent_config.keys():
                    agent_config[key] = agent_config[agent_config[key]]
                    
            agent_configs.append(agent_config)

    return agent_configs

def run_experiment(
        n_agents,
        env_config,
        training_targets_set,
        n_training_target_repeats,
        test_combs_set,
        load_agent_configs = False,
        agent_configs_path = None,
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
    model_configs : list
        A list of basic model configurations.
    output_path : str
        The path to save the data. If left as False, will not save data.

    Returns
    -------
    None
    """

    np.random.seed(seed)

    # Create the directories to save data to
    if output_path:
        for model_config in model_configs:
            model_label = model_config['model_label']
            makedirs(f'{output_path}/{model_label}/training', exist_ok=True)
            makedirs(f'{output_path}/{model_label}/test', exist_ok=True)
            makedirs(f'{output_path}/{model_label}/representations', exist_ok=True)

    # Load all agent configurations
    if load_agent_configs:
        agent_configs = []
        for subj in listdir(agent_configs_path):
            if subj == '.DS_Store': continue
            agent_configs.extend(load_configs(f'{agent_configs_path}/{subj}'))

    # Generate all agent configurations
    else:
        agent_configs = generate_agent_configs(n_agents, model_configs)

    # Simulate all agents
    for agent_config in agent_configs:
        subj = agent_config['id']
        model_label = agent_config['model_label']
        print(f'Simulating - Agent: {subj}/{len(agent_configs)}, Model: {model_label}')
        
        # Simulate agent
        training_data, test_data, representations = simulate_agent(
            agent_config,
            env_config,
            training_targets_set,
            n_training_target_repeats,
            test_combs_set
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
        for key in [*agent_config][::-1]:
            training_df.insert(0, key, agent_config[key])
            test_df.insert(0, key, agent_config[key])

        # Save data 
        if output_path:
            model_path = f'{output_path}/{model_label}'
            training_df.to_csv(
                f'{model_path}/training/training_{subj}.csv',
                index=False
            )
            test_df.to_csv(
                f'{model_path}/test/test_{subj}.csv',
                index=False
            )
            with open(
                f'{model_path}/representations/representations_{subj}.pkl',
                'wb'
            ) as file:
                pickle.dump(representations, file)