#!/bin/bash

# Interrompe o script se houver erro em qualquer comando
set -e

MODEL_NAME=${1:-CTGAN}
RUN_TYPE_NAME=${2:-reference}

# Pastas/datasets a serem ignorados por padrão
DEFAULT_EXCLUDES=()

# Exemplo: ./synthetic_run_all_datasets.sh TVAE reference "cidds,breast_cancer"
EXCLUDE_PARAM=${3:-""}
IFS=',' read -r -a CLI_EXCLUDES <<< "$EXCLUDE_PARAM"
EXCLUDED_DATASETS=("${DEFAULT_EXCLUDES[@]}" "${CLI_EXCLUDES[@]}")

is_excluded() {
    local target="$1"
    for item in "${EXCLUDED_DATASETS[@]}"; do
        # Remove espaços em branco nas pontas
        item="${item#"${item%%[![:space:]]*}"}"
        item="${item%"${item##*[![:space:]]}"}"
        if [ -n "$item" ] && [ "$item" = "$target" ]; then
            return 0
        fi
    done
    return 1
}

# Verifica se o diretório base existe antes de tentar rodar
if [ ! -d "results/$MODEL_NAME" ]; then
    echo "Erro: O diretório 'results/$MODEL_NAME' não foi encontrado."
    exit 1
fi

for dataset_dir in "results/$MODEL_NAME"/*/; do
    [ -d "$dataset_dir" ] || continue
    DATASET_NAME=$(basename "$dataset_dir")

    if is_excluded "$DATASET_NAME"; then
        echo "Pulando dataset excluído: $DATASET_NAME"
        continue
    fi

    echo "Iniciando experimento parte 1: gerando os samples $DATASET_NAME"
    if [ ! -d "results/$MODEL_NAME/$DATASET_NAME/$RUN_TYPE_NAME" ]; then
        echo "Erro: O diretório 'results/$MODEL_NAME/$DATASET_NAME/$RUN_TYPE_NAME' não foi encontrado."
        exit 1
    fi

    CONFIG_PATH="results/$MODEL_NAME/$DATASET_NAME/$RUN_TYPE_NAME/synthetic/config.toml"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Erro: O arquivo de configuração '$CONFIG_PATH' não foi encontrado."
        exit 1
    fi

    echo "conda run --no-capture-output -n SynthEnvironment python ./experiments_pipes/synthetic_generator_pipe.py --config $CONFIG_PATH"
    conda run --no-capture-output -n SynthEnvironment python ./experiments_pipes/synthetic_generator_pipe.py --config "$CONFIG_PATH"
done


echo "Todos os testes foram concluídos com sucesso!"