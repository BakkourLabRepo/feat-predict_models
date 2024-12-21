import argparse
import importlib
from scripts.simulate_funcs import run_experiment

def import_config(config_fname):
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
        config = importlib.import_module(f'configs.{config_module_name}')
        return config.experiment_config
    except ModuleNotFoundError:
        print(f"Error: {config_module_name}.py not found.")

def main():

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description = 'Specify which config file to import.'
        )
    parser.add_argument(
        'config_fname',
        type = str,
        help = "Specify the file name for the config to import."
    )
    config_fname = parser.parse_args().config_fname

    # Import the experiment configuration
    experiment_config = import_config(config_fname)

    # Run the experiment
    run_experiment(**experiment_config)

if __name__ == "__main__":
    main()

