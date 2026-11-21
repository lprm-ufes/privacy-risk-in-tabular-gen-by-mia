import toml
import pandas as pd
from pathlib import Path

def read_info(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except toml.TomlDecodeError as e:
        print(f"Error decoding TOML: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

REAL_FOLDER = "./real"
REFERENCE_FOLDER = "./reference"
DATASET_NAME = "cidds"

info = read_info('./info.toml')

full_df = pd.read_csv(f"./{DATASET_NAME}.csv")

use_columns = info['split-info']['use_col']

full_df = full_df[use_columns]

# Separando conjunto real e de referência
real_full_df = full_df.sample(frac=0.60, random_state=42)

reference_full_df = full_df.drop(real_full_df.index)

# Separando conjunto real em treino e teste
real_train_df = real_full_df.sample(frac=0.75, random_state=42)

real_test_df = real_full_df.drop(real_train_df.index)

# Salvando os dados
real_folder = Path(REAL_FOLDER)
reference_folder = Path(REFERENCE_FOLDER)

real_folder.mkdir(parents=True, exist_ok=True)
reference_folder.mkdir(parents=True, exist_ok=True)

real_train_df.to_csv(real_folder / f"{DATASET_NAME}_train.csv", index=False)
real_test_df.to_csv(real_folder / f"{DATASET_NAME}_test.csv", index=False)

reference_full_df.to_csv(reference_folder / f"{DATASET_NAME}_reference.csv", index=False)