import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
from utils import utils
import toml
import pandas as pd

from tabular_generative_models import get_model_generator


def _get_next_hiperparams(keys, currKey, hiperparams_grid, final_params):
    if currKey >= len(keys):
        yield final_params.copy()
        return

    for param in hiperparams_grid[keys[currKey]]:
        final_params[keys[currKey]] = param
        yield from _get_next_hiperparams(keys, currKey + 1, hiperparams_grid, final_params)


def get_next_hiperparams(hiperparams_grid):
    all_keys = list(hiperparams_grid.keys())
    yield from _get_next_hiperparams(all_keys, 0, hiperparams_grid, {})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Caminho para o arquivo de configuração.')
    args = parser.parse_args()

    config = utils.read_config(args.config)

    real_dataset_input = Path(config['io']['real_dataset_input'])
    if not real_dataset_input.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {real_dataset_input}")

    tracker_output_folder = Path(config['io']['tracker_file_output_folder'])
    model_output_folder = Path(config['io']['results_output_model_folder'])
    dataset_output_folder = Path(config['io']['results_output_synthetic_folder'])

    tracker_output_folder.mkdir(parents=True, exist_ok=True)
    dataset_output_folder.mkdir(parents=True, exist_ok=True)

    save_models = config['exp'].get('save_models', True)
    if save_models:
        model_output_folder.mkdir(parents=True, exist_ok=True)

    X_full = pd.read_csv(real_dataset_input)
    n_total = len(X_full)

    synthetic_size = config['exp'].get('synthetic_size', n_total)
    start_exp = config['exp'].get('start_exp', None)
    end_exp = config['exp'].get('end_exp', None)

    hyperparameters_grid = config['model_hyperparameters']
    exp_seeds = config['exp_params']['seeds']

    tracker_results = []
    tracker_file_path = tracker_output_folder / 'synthetic_tracker.csv'

    dataset_info = toml.load(config['io']['dataset_info'])
    num_columns = dataset_info['columns']['num_col']
    cat_columns = dataset_info['columns']['cat_col']
    target_column = dataset_info['columns']['label']

    generator = get_model_generator(
        model_name=config['exp']['model'],
        data=X_full,
        num_cols=num_columns,
        cat_cols=cat_columns,
        target_col=target_column,
    )

    tracker_df = None
    if tracker_file_path.is_file():
        tracker_df = pd.read_csv(tracker_file_path, encoding='utf-8')

    exp_idx = 0
    for seed in exp_seeds:
        for params in get_next_hiperparams(hyperparameters_grid):
            exp_idx = exp_idx + 1
            if start_exp is not None and exp_idx < start_exp:
                continue
            if end_exp is not None and exp_idx > end_exp:
                continue

            synthetic_file = f"dataset_{exp_idx}.csv"
            model_file = ""
            if save_models:
                model_file = f"model_{exp_idx}{generator.file_extension}"

            print(f"\n[Exp {exp_idx}] Seed: {seed} Parâmetros: {params}")
            X_synthetic, train_time, generate_time = generator.fit_and_generate(
                params=params,
                count=synthetic_size,
                seed=seed,
            )

            X_synthetic.to_csv(dataset_output_folder / synthetic_file, index=False)
            print(f"Dataset salvo em: {dataset_output_folder / synthetic_file}")

            if save_models:
                generator.save(model_output_folder / model_file)
                print(f"Modelo salvo em: {model_output_folder / model_file}")

            row_details = {"exp_idx": exp_idx}
            row_details.update(params)
            row_details.update({
                "train_time": train_time,
                "generate_time": generate_time,
                "dataset_file": synthetic_file,
                "model_file": model_file,
            })

            if tracker_df is not None:
                tracker_df = pd.concat([tracker_df, pd.DataFrame([row_details])])
            else:
                tracker_df = pd.DataFrame([row_details])

            tracker_df.to_csv(tracker_file_path, index=False, encoding='utf-8')

    print("\nExecução finalizada.")