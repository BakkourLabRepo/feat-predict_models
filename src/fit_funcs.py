import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.Env import Env

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
    converted_state_arrat = []
    for state_str in state_array:
        state = convert_state_str(state_str)
        if len(feature_reorder) > 0:
            state = state[feature_reorder]
        converted_state_arrat.append(state)
    return converted_state_arrat

def get_options_step(env, target, n_step_inference):
    """
    Get the step at which to sample options based on the target step
    and the number of steps to infer composition from.

    Arguments
    ---------
    env : Env
        Instance of the environment.
    target : numpy.ndarray
        The target state.
    n_step_inference : int or None
        Number of steps to infer composition from. If None, will use
        env.max_steps.
    
    Returns
    -------
    options_step : int
        The step at which to sample options.
    """

    # Get options step based on target step
    target_step = env.check_step(target)
    options_step = target_step - n_step_inference

    # Can't have step before initial step
    if options_step < 0:
        options_step = 0

    return options_step

def train_agent(agent, env, data, n_step_inference=None):
    """
    Trains the agent on the training phase.

    Arguments
    ---------
    agent : SuccessorFeatures
        The agent to train.
    env : Env
        The environment to train the agent in.
    data : pandas.DataFrame
        Training data.
    n_step_inference : int or None
        Number of steps to infer composition from. If None, will use
        env.max_steps.

    Returns
    -------
    probs : numpy.ndarray
        Array of choice probabilities.
    """

    # If n_step_inference not specified, use max steps - 1
    if n_step_inference is None:
        n_step_inference = env.max_steps

    probs = []
    for t in range(len(data)):

        # Get trial information
        target = data.loc[t, 'target']
        options_comb = data.loc[t, 'options_comb']
        composition = data.loc[t, 'composition']

        # Set target as task
        agent.set_task(target)

        # Get options step based on target step
        options_step = get_options_step(env, target, n_step_inference)

        # Generate feature set
        env.sample_features(comb=options_comb, step=options_step)

        # Get composition
        p = agent.compose_from_set(env.a, set_composition=composition)[1]
        probs.append(p)
        env.s = composition
        agent.update_memory(env.s)

        # Step environment
        step = 0
        while True:
            step += 1
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

            # Terminate if max steps reached
            if step >= env.max_steps:

                # For terminal state, include absorbing transition
                if env.check_terminal(env.s):
                    step -= 1

                else: 
                    break 

    probs = np.array(probs)
    return probs

def test_agent(agent, env, data, n_step_inference=None):
    """
    Test the agent on the test phase.

    Arguments
    ---------
    agent : SuccessorFeatures
        The agent to train.
    env : Env
        The environment to train the agent in.
    data : pandas.DataFrame
        Test data.
    n_step_inference : int or None
        Number of steps to infer composition from. If None, will use

    Returns
    -------
    probs : numpy.ndarray
        Array of choice probabilities.
    """

    # If n_step_inference not specified, use max steps - 1
    if n_step_inference is None:
        n_step_inference = env.max_steps

    probs = []
    for t in range(len(data)):

        # Get trial information
        target = data.loc[t, 'target']
        options_comb = data.loc[t, 'options_comb']
        composition = data.loc[t, 'composition']

        # Set target as task
        agent.set_task(target)
        
        # Get options step based on target step
        options_step = get_options_step(env, target, n_step_inference)

        # Generate feature set
        env.sample_features(comb=options_comb, step=options_step)

        # Get composition
        p = agent.compose_from_set(env.a, set_composition=composition)[1]
        probs.append(p)

    probs = np.array(probs)
    return probs

def likfun(
        params,
        data,
        Model,
        agent_config,
        env_config,
        params_to_fit,
        running_agent
    ):
    """
    Calculate the negative log likelihood (nLL) of the model.

    Arguments
    ---------
    params : list
        Parameter values for this fitting iteration.
    data : dict
        Dictionary containing training and test data.
    Model : class
        The model class to instantiate.
    agent_config : dict
        Dictionary containing the agent configuration.
    env_config : dict
        Dictionary containing the environment configuration.
    params_to_fit : list
        A list of the parameter names for params.
    running_agent : dict
        A dictionary containing the agent object. Used to track the
        agent with the best fit.

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
    agent = Model(env, **this_agent_config)

    # Get action probabilities
    training_probs = train_agent(agent, env, data['training'])
    test_probs = test_agent(agent, env, data['test'])
    probs = np.concatenate([training_probs, test_probs])

    if np.any(np.isnan(probs)):
        print('NaN in probs!')

    if np.any(probs == 0):
        probs[probs == 0] = 1e-10

    # Calculate negative log likelihood
    nLL = probs_to_nll(probs)

    # Track agent for best fit
    running_agent['agent'] = agent

    return nLL


def fit_model(
        data,
        Model,
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
    Model : class
        The model class to instantiate.
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
    null_result : dict
        The null model results.
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
            self.agent = None

    best_result = BestResult(params_to_fit)
    
    # Fit model with multiple random starts
    unchanged_count = 0
    for start in range(n_starts):

        # Random initialization
        x0 = []
        for param in params_to_fit:
            if (
                (param == 'beta') or
                (param == 'beta_test') or
                (param == 'alpha_decay')
            ):
                x0.append(1/np.random.uniform(0, 1) - 1)
            elif param == 'sampler_specificity':
                x0.append(1/np.random.uniform(0, 1))
            else:
                x0.append(np.random.uniform(0, 1))

        # Fit model
        running_agent = {'agent': None}
        result = minimize(
            likfun,
            x0,
            args = (
                data,
                Model,
                agent_config,
                env_config,
                params_to_fit,
                running_agent
                ),
            method = 'L-BFGS-B',
            bounds = bounds
        )

        # Update best result
        if result.fun < best_result.fun:
            best_result = result
            best_result.agent = running_agent['agent']
        elif result.success:
            unchanged_count += 1

        # Break if no improvement in fit after some number of starts
        if unchanged_count >= max_unchanged:
            break
    best_result.n_starts = start + 1

    # Get null nLL and AIC
    null_probs = [.25]*len(data['training']) + [.25]*len(data['test'])
    null_nll = probs_to_nll(null_probs)
    null_result = {
        'nll': null_nll,
        'aic': nll_to_aic(null_nll, 0)
    }
    best_result.aic = nll_to_aic(best_result.fun, len(params_to_fit))

    # Construct agent config with fit parameters
    fit_agent_config = agent_config.copy()
    for i, param in enumerate(params_to_fit):
        fit_agent_config[param] = best_result.x[i]
    for key in agent_config.keys():
        if agent_config[key] in agent_config.keys():
            fit_agent_config[key] = fit_agent_config[fit_agent_config[key]]

    return best_result, fit_agent_config, null_result

def fit_model_parallel(args):
    """
    Fits a model in parallel for a given subject and model configuration.

    Arguments
    ---------
    args : dict
        A dictionary containing the following keys:
        - 'subj' (str): Subject ID.
        - 'Model' (class): The model class to instantiate.
        - 'model_config' (dict): Model configuration.
        - 'data_path' (str): Path to the data.
        - 'env_config' (dict): Environment configuration.
        - 'parameter_bounds' (dict): Model parameters bounds.
        - 'n_starts' (int): The random starts for optimization.
        - 'max_unchanged' (int): Max iterations without improvement.
        - 'feature_reorder' (list): List of indices to reorder features
          in the between-feature transitions condition.

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

            # Check whether to re-order features for between condition
            this_feature_order = []
            if 'between_cond' in agent_data['training'].columns:
                condition = agent_data['training']['between_cond'].iloc[0]
                if (state_type == 'target') and (condition == 1):
                    this_feature_order = args['feature_reorder']
            
            # Perfom re-ordering
            agent_data[phase][state_type] = transform_state_array(
                agent_data[phase][state_type],
                feature_reorder = this_feature_order
            )

    # Fit this model
    result, fit_agent_config, null_result = fit_model(
        agent_data,
        args['Model'],
        args['model_config'],
        args['env_config'],
        parameter_bounds = args['parameter_bounds'],
        seed = subj,
        n_starts = args['n_starts'],
        max_unchanged = args['max_unchanged']
    )

    return result, fit_agent_config, null_result
