import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scripts.Successor_Features import Successor_Features
from scripts.Env import Env

def probs_to_nll(probs):
    """
    Convert choice probabilities to negative log likelihood.

    Arguments
    ---------
    probs : numpy.ndarray
        Array of choice probabilities.

    Returns
    -------
    nll : float
        Negative log likelihood.
    """
    nll = -np.sum(np.log(probs))
    return nll

def nll_to_aic(nll, n_params):
    """
    Convert negative log likelihood to Akaike Information Criterion (AIC).

    Arguments
    ---------
    nll : float
        Negative log likelihood.
    n_params : int
        Number of parameters in the model.

    Returns
    -------
    aic : float
        Akaike Information Criterion.
    """
    aic = 2*nll + 2*n_params
    return aic

def drop_missed_trials(data):
    """
    Drop trials with missing data.

    Arguments
    ---------
    data : pandas.DataFrame
        The data to clean.

    Returns
    -------
    data : pandas.DataFrame
        The cleaned data.
    """
    data = data.dropna(subset=['successor'])
    data = data.reset_index(drop=True)
    return data

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

def transform_state_array(state_array):
    """
    Applies convert_state_str to a list of state strings.

    Arguments
    ---------
    state_array : list
        A list of state strings.

    Returns
    -------
    state_array : list
        A list of converted state values.
    """
    state_array = [convert_state_str(state_str) for state_str in state_array]
    return state_array

def train_agent(agent, env, data):
    """
    Trains the agent on the training phase.

    Arguments
    ---------
    agent : Successor_Features
        The agent to train.
    env : Env
        The environment to train the agent in.
    data : pandas.DataFrame
        Training data.

    Returns
    -------
    probs : numpy.ndarray
        Array of choice probabilities.
    """

    probs = []
    for t in range(len(data)):

        # Get trial information
        target = data.loc[t, 'target']
        options_comb = data.loc[t, 'options_comb']
        composition = data.loc[t, 'composition']

        # Set target as task
        agent.set_task(target)

        # Generate feature set
        env.sample_features(
            comb = options_comb,
            terminal = False
        )

        # Get composition
        p = agent.compose_from_set(env.a, set_composition=composition)[1]
        probs.append(p)
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

    probs = np.array(probs)
    return probs

def test_agent(agent, env, data):
    """
    Test the agent on the test phase.

    Arguments
    ---------
    agent : Successor_Features
        The agent to train.
    env : Env
        The environment to train the agent in.
    data : pandas.DataFrame
        Test data.

    Returns
    -------
    probs : numpy.ndarray
        Array of choice probabilities.
    """
    probs = []
    for t in range(len(data)):

        # Get trial information
        target = data.loc[t, 'target']
        options_comb = data.loc[t, 'options_comb']
        composition = data.loc[t, 'composition']

        # Set target as task
        agent.set_task(target)

        # Generate feature set
        env.sample_features(
            options_comb,
            terminal = False
        )

        # Get composition
        p = agent.compose_from_set(env.a, set_composition=composition)[1]

    probs = np.array(probs)
    return probs

def likfun(
        params,
        data,
        agent_config,
        env_config,
        params_to_fit
    ):
    """
    Calculate the negative log likelihood (nLL) of the model.

    Arguments
    ---------
    params : list
        Parameter values for this fitting iteration.
    data : dict
        Dictionary containing training and test data.
    agent_config : dict
        Dictionary containing the agent configuration.
    env_config : dict
        Dictionary containing the environment configuration.
    params_to_fit : list
        A list of the parameter names for params.

    Returns
    -------
    nLL: float
        Negative log likelihood of the model given the data.
    """

    # Set parameters
    this_agent_config = agent_config.copy()
    for j, param in enumerate(params_to_fit):
        this_agent_config[param] = params[j]
    for key in this_agent_config.keys():
        if this_agent_config[key] in this_agent_config.keys():
            this_agent_config[key] = this_agent_config[this_agent_config[key]]

    # Initialize environment and agent
    env = Env(**env_config)
    agent = Successor_Features(env, **this_agent_config)

    # Get action probabilities
    training_probs = train_agent(agent, env, data['training'])
    agent.beta = agent.beta_test
    test_probs = test_agent(agent, env, data['test'])
    probs = np.concatenate([training_probs, test_probs])

    # Calculate negative log likelihood
    if np.any(np.isnan(probs)):
        print(probs)
    nLL = probs_to_nll(probs)

    return nLL


def fit_model(
        data,
        agent_config,
        env_config,
        parameter_bounds = None,
        seed = False,
        n_starts = 10,
        max_unchanged = 5
    ):
    """
    Fit the successor features model.

    Arguments
    ---------
    data : dict
        Dictionary containing training and test data.
    agent_config : dict
        Dictionary containing the agent configuration.
    env_config : dict
        Dictionary containing the environment configuration.
    parameter_bounds : dict
        Dictionary containing the bounds for all parameters.
    seed : int
        Seed for random number generation.
    n_starts : int
        Number of random initializations for the optimizer
    max_unchanged : int
        The maximum number of starts without improvement in fit (default: 5).

    Returns
    -------
    best_result : object
        The best result from fitting the model.
    fit_agent_config : dict
        The agent configuration with the fit parameters.
    """

    if seed:
        np.random.seed(seed)

    # Get parameters to fit
    agent_config['id'] = data['training']['id'].iloc[0]
    params_to_fit = []
    for param in agent_config:
        if agent_config[param] == None:
            params_to_fit.append(param)

    # Set bounds for parameters to fit
    if parameter_bounds == None:
        bounds = None
    else:
        bounds = []
        for param in params_to_fit:
            bounds.append(parameter_bounds[param])

    # Init object for tracking the best fit across starts
    class BestResult:
        def __init__(self, params_to_fit):
            self.fun = np.inf
            self.success = False
            self.x = [np.nan]*len(params_to_fit)

    best_result = BestResult(params_to_fit)
    
    # Fit model with multiple random starts
    unchanged_count = 0
    for start in range(n_starts):

        # Random initialization
        x0 = []
        for param in params_to_fit:
            if (param == 'beta') or (param == 'beta_test'):
                x0.append(1/np.random.uniform(0, 1))
            elif param == 'sampler_specificity':
                x0.append(1 + 1/np.random.uniform(0, 1))
            else:
                x0.append(np.random.uniform(0, 1))

        # Fit model
        result = minimize(
            likfun,
            x0,
            args = (data, agent_config, env_config, params_to_fit),
            method = 'L-BFGS-B',
            bounds = bounds
        )

        # Update best result
        if result.fun < best_result.fun:
            best_result = result
        else:
            unchanged_count += 1

        # Break if no improvement in fit after some number of starts
        if unchanged_count >= max_unchanged:
            break
    best_result.n_starts = start + 1

    # Get null nLL and AIC
    null_probs = [.5]*len(data['training']) + [.5]*len(data['test'])
    best_result.null_nll = probs_to_nll(null_probs)
    best_result.aic = nll_to_aic(best_result.fun, len(params_to_fit))

    # Construct agent config with fit parameters
    fit_agent_config = agent_config.copy()
    for i, param in enumerate(params_to_fit):
        fit_agent_config[param] = best_result.x[i]
    for key in agent_config.keys():
        if agent_config[key] in agent_config.keys():
            fit_agent_config[key] = fit_agent_config[fit_agent_config[key]]

    return best_result, fit_agent_config

def fit_model_parallel(args):
    """
    Fits a model in parallel for a given subject and model configuration.

    Arguments
    ---------
    args : dict
        A dictionary containing the following keys:
        - 'subj' (str): Subject ID.
        - 'model_config' (dict): Model configuration.
        - 'data_path' (str): Path to the data.
        - 'env_config' (dict): Environment configuration.
        - 'parameter_bounds' (dict): Model parameters bounds.
        - 'n_starts' (int): The random starts for optimization.
        - 'max_unchanged' (int): Max iterations without improvement.

    Returns
    -------
    result : object
        Model fitting results
    fit_agent_config : dict
        The agent configuration with the fit parameters.
    """

    subj = args['subj']
    model_label = args['model_config']['model_label']
    data_path = args['data_path']
    print(f'Fitting - Subject: {subj}, Model: {model_label}')

    # Load data
    agent_data = {
        'training': pd.read_csv(f'{data_path}/training/training_{subj}.csv'),
        'test': pd.read_csv(f'{data_path}/test/test_{subj}.csv')
    }
    agent_data['training'] = drop_missed_trials(agent_data['training'])
    agent_data['test'] = drop_missed_trials(agent_data['test'])

    # Convert state strings to arrays
    for phase in agent_data.keys():
        for state_type in ['target', 'options_comb', 'composition']:
            agent_data[phase][state_type] = transform_state_array(
                agent_data[phase][state_type]
            )

    # Fit this model
    result, fit_agent_config = fit_model(
        agent_data,
        args['model_config'],
        args['env_config'],
        parameter_bounds = args['parameter_bounds'],
        seed = subj,
        n_starts = args['n_starts'],
        max_unchanged = args['max_unchanged']
    )

    return result, fit_agent_config
