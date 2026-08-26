from os import listdir
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse.csgraph import shortest_path
from src.Env import Env
from src.SuccessorFeatures import SuccessorFeatures
from src.MBRL import MBRL

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
    data = data[data['successor'] != '[]']
    data = data.reset_index(drop=True)
    return data

def get_step(start_state, shortest_paths):
    """
    Get the step of a given state based on the shortest paths in the
    environment. The max step is the longest shortest path based on 
    the instance transition matrix.

    Arguments
    ---------
    start_state : str
        The starting state.
    shortest_paths : ndarray
        The shortest paths in the environment.

    Returns
    -------
    step : int
        The step of the given state.
    """
    start_state = convert_state_str(start_state)
    example_inst = start_state[start_state != 0][0] - 1
    max_step = int(np.max(shortest_paths))
    step = int(max_step - np.max(shortest_paths[example_inst]) + 1)
    return step

def add_step_info(data, tmat):
    """
    Add step information to the data based on the shortest paths in 
    the environment. The max step is the longest shortest path based on
    the instance transition matrix.

    Arguments
    ---------
    data : pandas.DataFrame
        The data to add step information to.
    tmat : ndarray
        The instance transition matrix of the environment.
    
    Returns
    -------
    data : pandas.DataFrame
        The update data frame with step information added.
    """
    

    # Compute shortest paths to identify inference step
    shortest_paths = shortest_path(tmat, directed=True)
    shortest_paths[shortest_paths == np.inf] = 0

    # Add step information to data
    data['step'] = data['correct_composition'].apply(
        get_step,
        args = (shortest_paths,)
        )
    
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
    state_str = state_str.replace(',', ' ')
    state_str = state_str[1:-1].split(' ')
    state_arr = np.array(state_str, dtype=int)
    return state_arr

def recode_state_array(
        state_arr,
        start_step_arr = None,
        n_steps_arr = None,
        feature_reorder = []
        ):
    """
    Recode an array of states based on the feature transition strucutre

    Arguments
    ---------
    state_arr : numpy.ndarray
        The array of states to recode.
    start_step_arr : numpy.ndarray or None
        An array containing the starting step for each state. 
    n_steps_arr : numpy.ndarray or None
        An array containing the number of steps inferences are made
        over from that state.
    feature_reorder : list
        A list of indices to reorder features based on at each step.

    Returns
    -------
    recoded_state_arr : numpy.ndarray
        The recoded array of states.
    """
    if len(feature_reorder) == 0:
        return state_arr
    start_step = 1
    n_steps = 1
    recoded_state_arr = np.copy(state_arr)
    for i, state in enumerate(state_arr):
        if start_step_arr is not None:
            start_step = start_step_arr[i]
            n_steps = n_steps_arr[i]
        recoded_state = np.copy(state)
        for step in range(start_step, start_step + n_steps):
            recoded_state = recoded_state[feature_reorder[step - 1]]
        recoded_state_arr[i] = recoded_state
    return recoded_state_arr

def train_agent(agent, env, data):
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
        step = data.loc[t, 'step'] - 1

        # Set target as task
        agent.set_task(target)

        # Generate feature set
        env.sample_features(comb=options_comb, step=step)

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

def test_agent(agent, env, data):
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
        step = data.loc[t, 'step'] - 1

        # Set target as task
        agent.set_task(target)

        # Generate feature set
        env.sample_features(comb=options_comb, step=step)

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
        - 'model' (str): String for the model class to instantiate.
        - 'model_config' (dict): Model configuration.
        - 'data_path' (str): Path to the data.
        - 'bids' (bool): Whether the data is in BIDS format.
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
    model = args['model']
    model_label = args['model_config']['model_label']
    data_path = args['data_path']
    print(f'Fitting - Subject: {subj}, Model: {model_label}')

    # Load data
    if args['bids']:
        agent_data = {'training': [], 'test': []}
        for phase in ['training', 'test']:
            for fname in listdir(f'{data_path}/sub-{subj}'):
                if (phase in fname) and (fname.endswith('.csv')):
                    agent_data[phase].append(
                        pd.read_csv(f'{data_path}/sub-{subj}/{fname}')
                    )
                    
            agent_data[phase] = pd.concat(agent_data[phase], ignore_index=True)
    else:
        try:
            agent_data = {
                'training': pd.read_csv(f'{data_path}/training/training_{subj}.csv'),
                'test': pd.read_csv(f'{data_path}/test/test_{subj}.csv')
            }
        except:
            agent_data = {
                'training': pd.read_csv(f'{data_path}/training/sub-{subj}_task-training.csv'),
                'test': pd.read_csv(f'{data_path}/test/sub-{subj}_task-test.csv')
        }
    agent_data['training'] = drop_missed_trials(agent_data['training'])
    agent_data['test'] = drop_missed_trials(agent_data['test'])

    if 'n_steps' not in agent_data['training'].columns:
        agent_data['training']['n_steps'] = 1
        agent_data['training']['step'] = 1
    else:
        agent_data['training'] = add_step_info(
            agent_data['training'],
            args['env_config']['tmat']
            )
    if 'n_steps' not in agent_data['test'].columns:
        agent_data['test']['n_steps'] = 1
        agent_data['test']['step'] = 1
    else:
        agent_data['test'] = add_step_info(
            agent_data['test'],
            args['env_config']['tmat']
            )

    # Convert state strings to arrays
    for phase in agent_data.keys():
        for state_type in ['target', 'options_comb', 'composition']:

            # Check whether to re-order features for between condition
            this_feature_order = []
            if 'condition' in agent_data[phase].columns:
                condition = agent_data[phase]['condition'].iloc[0]
                if state_type == 'target':
                    this_feature_order = args['feature_reorder'][condition]
            elif 'between_cond' in agent_data[phase].columns:
                condition = agent_data[phase]['between_cond'].iloc[0]
                if state_type == 'target':
                    this_feature_order = args['feature_reorder'][condition]

            # Transform from strings to array
            agent_data[phase][state_type] = agent_data[phase][state_type].apply(
                convert_state_str
            )
            
            # Perfom re-ordering 
            agent_data[phase][state_type] = recode_state_array(
                agent_data[phase][state_type].values,
                start_step_arr = agent_data[phase]['step'].values,
                n_steps_arr = agent_data[phase]['n_steps'].values,
                feature_reorder = this_feature_order,
            )

    # Select model class
    if model == 'MBRL':
        Model = MBRL
    else:
        Model = SuccessorFeatures
    
    # Fit this model
    result, fit_agent_config, null_result = fit_model(
        agent_data,
        Model,
        args['model_config'],
        args['env_config'],
        parameter_bounds = args['parameter_bounds'],
        seed = subj,
        n_starts = args['n_starts'],
        max_unchanged = args['max_unchanged']
    )

    return result, (model, fit_agent_config), null_result
