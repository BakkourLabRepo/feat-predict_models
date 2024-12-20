from scripts.simulate_funcs import run_experiment
from scripts.simulate_config import (
    N_AGENTS,
    ENV_CONFIG,
    TRAINING_TARGETS_SET,
    N_TRAINING_TARGET_REPEATS,
    TEST_COMBS_SET,
    FIXED_TRAINING,
    AGENT_CONFIGS_PATH,
    TRAINING_TRIAL_INFO_PATH,
    TEST_TRIAL_INFO_PATH,
    MATCH_TRIALS_TO_AGENTS,
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
    fixed_training = FIXED_TRAINING,
    agent_configs_path = AGENT_CONFIGS_PATH,
    training_trial_info_path = TRAINING_TRIAL_INFO_PATH,
    test_trial_info_path = TEST_TRIAL_INFO_PATH,
    match_trials_to_agents = MATCH_TRIALS_TO_AGENTS,
    model_configs = MODEL_CONFIGS,
    output_path = OUTPUT_PATH,
    seed = SEED
)