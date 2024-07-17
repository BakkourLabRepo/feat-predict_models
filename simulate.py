from scripts.simulate_funcs import run_experiment
from scripts.simulate_config import (
    N_AGENTS,
    ENV_CONFIG,
    TRAINING_TARGETS_SET,
    N_TRAINING_TARGET_REPEATS,
    TEST_COMBS_SET,
    LOAD_AGENT_CONFIGS,
    AGENT_CONFIGS_PATH,
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
    load_agent_configs = LOAD_AGENT_CONFIGS,
    agent_configs_path = AGENT_CONFIGS_PATH,
    model_configs = MODEL_CONFIGS,
    output_path = OUTPUT_PATH,
    seed = SEED
)