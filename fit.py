import pandas as pd
from os import listdir, makedirs
from re import search
import concurrent.futures
import pickle
from scripts.fit_funcs import fit_model_parallel
from scripts.fit_config import (
    DATA_PATH,
    RESULTS_PATH,
    RESULTS_FNAME,
    N_STARTS,
    MAX_UNCHANGED,
    OVERWRITE,
    NUM_CORES,
    MODEL_CONFIGS,
    PARAMETER_BOUNDS,
    ENV_CONFIG,
    FEATURE_REORDER
)


# Get subject IDs
fnames = listdir(f'{DATA_PATH}/training')
subj_ids = sorted([int(search('\d+', f)[0]) for f in fnames])

# Make results directory if it does not exist
for subj in subj_ids:
    makedirs(f'{RESULTS_PATH}/fit_agent_configs/{subj}', exist_ok=True)

# Load existing results
try:
    results = pd.read_csv(f'{RESULTS_PATH}/{RESULTS_FNAME}.csv')

    # Only fit for subjects and models with no results
    if OVERWRITE:
        completed_fits = []
    else:
        completed_fits = (
            results['id'].astype(str) +
            results['model_label']
        ).values

# Initialize empty data frame 
except:
    results = pd.DataFrame(columns=[
        'id',
        'model_label',
        'success',
        'n_starts',
        'nll',
        'aic',
        'null_nll',
        'alpha',
        'beta',
        'beta_test',
        'segmentation',
        'conjunctive_starts',
        'conjunctive_successors',
        'conjunctive_composition',
        'memory_sampler',
        'sampler_feature_weight',
        'sampler_recency_weight',
        'sampler_specificity'
        ])
    completed_fits = []

# Generate fitting arguments
fitting_args = []
for subj in subj_ids:
    for model_config in MODEL_CONFIGS:

        # Skip if already fit
        model_label = model_config['model_label']
        if str(subj) + model_label in completed_fits:
            continue

        fitting_args.append({
            'subj': subj,
            'model_config': model_config,
            'data_path': DATA_PATH,
            'env_config': ENV_CONFIG,
            'parameter_bounds': PARAMETER_BOUNDS,
            'n_starts': N_STARTS,
            'max_unchanged': MAX_UNCHANGED,
            'feature_reorder': FEATURE_REORDER.copy()
        })

# Run paralellized fitting prcedure
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=NUM_CORES
        ) as executor:

        # Submit fitting jobs
        futures = {
            executor.submit(fit_model_parallel, arg):
            arg for arg in fitting_args
        }
        
        # Save results as they are produced
        for future in concurrent.futures.as_completed(futures):
            this_result, this_agent_config = future.result()
            
            # Export results to .csv
            subj = this_agent_config['id']
            results = pd.concat([results, pd.DataFrame([{
                'id': subj,
                'success': this_result.success,
                'n_starts': this_result.n_starts,
                'nll': this_result.fun,
                'aic': this_result.aic,
                'null_nll': this_result.null_nll,
                **this_agent_config
            }])], ignore_index=True)
            results = results.sort_values(by=['id', 'model_label'])
            results.to_csv(f'{RESULTS_PATH}/{RESULTS_FNAME}.csv', index=False)

            # Export agent config to pickle
            dpath = f'{RESULTS_PATH}/fit_agent_configs/{subj}'
            fname = f'{subj}_{this_agent_config["model_label"]}.pkl'
            with open(f'{dpath}/{fname}', 'wb') as f:
                pickle.dump(this_agent_config, f)




