import importlib

def import_config(config_fname, configs_dir='configs'):
    """
    Import the experiment configuration from the specified file.
    
    Arguments
    ---------
    config_fname : str
        The name of the configuration file to import. Should be in the
        configs directory.
    
    Returns
    -------
    dict
        The experiment configuration.
    """
    config_fname = config_fname.replace('.py', '')
    config_module_name = f"{config_fname}"
    try:
        config = importlib.import_module(f'{configs_dir}.{config_module_name}')
        return config.experiment_config
    except ModuleNotFoundError:
        print(f"Error: {config_module_name}.py not found.")