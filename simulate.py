import argparse
from src.simulate_funcs import run_experiment
from src.utils import import_config

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
    experiment_config = import_config(
        config_fname,
        configs_dir = 'configs.simulate'
        )

    # Run the experiment
    for key in experiment_config:
        run_experiment(**experiment_config[key])

if __name__ == "__main__":
    main()

