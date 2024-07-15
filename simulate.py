from scripts.simulate_funcs import run_experiment
from scripts.simulate_config import (
    N_AGENTS,
    ENV_CONFIG,
    TRAINING_TARGETS_SET,
    N_TRAINING_TARGET_REPEATS,
    TEST_COMBS_SET,
    MODEL_CONFIGS,
    OUTPUT_PATH,
    SEED
)

# Run the experiment
run_experiment(
    N_AGENTS,
    ENV_CONFIG,
    TRAINING_TARGETS_SET,
    N_TRAINING_TARGET_REPEATS,
    TEST_COMBS_SET,
    MODEL_CONFIGS,
    output_path = OUTPUT_PATH,
    seed = SEED
)