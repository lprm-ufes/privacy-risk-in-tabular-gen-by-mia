#!/bin/bash

# Interrompe o script se houver erro em qualquer comando
set -e

# Diretório base dos experimentos de domain drift / domain shift
BASE_DIR=${BASE_DIR:-"results_domain_shift"}
if [ ! -d "$BASE_DIR" ] && [ -d "results_domain_shift" ]; then
    BASE_DIR="results_domain_shift"
fi

MODEL_NAME=${1:-all}
RUN_TYPE_NAME=${2:-reference}

# Pastas/datasets a serem ignorados por padrão
DEFAULT_EXCLUDES=()

# Exemplo: ./domain_drift_synth.sh all reference "breast_cancer"
# Ou para um modelo específico: ./domain_drift_synth.sh CTGAN
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
if [ ! -d "$BASE_DIR" ]; then
    echo "Erro: O diretório '$BASE_DIR' não foi encontrado."
    exit 1
fi

# Determina quais modelos serão executados
if [ "$MODEL_NAME" = "all" ] || [ -z "$MODEL_NAME" ]; then
    MODELS=()
    for model_dir in "$BASE_DIR"/*/; do
        [ -d "$model_dir" ] || continue
        MODELS+=("$(basename "$model_dir")")
    done
else
    if [ ! -d "$BASE_DIR/$MODEL_NAME" ]; then
        echo "Erro: O diretório '$BASE_DIR/$MODEL_NAME' não foi encontrado."
        exit 1
    fi
    MODELS=("$MODEL_NAME")
fi

for current_model in "${MODELS[@]}"; do
    for dataset_dir in "$BASE_DIR/$current_model"/*/; do
        [ -d "$dataset_dir" ] || continue
        DATASET_NAME=$(basename "$dataset_dir")

        if is_excluded "$DATASET_NAME"; then
            echo "Pulando dataset excluído: $DATASET_NAME ($current_model)"
            continue
        fi

        echo "Iniciando experimento parte 1: gerando os samples $DATASET_NAME para o modelo $current_model"
        if [ ! -d "$BASE_DIR/$current_model/$DATASET_NAME/$RUN_TYPE_NAME" ]; then
            echo "Erro: O diretório '$BASE_DIR/$current_model/$DATASET_NAME/$RUN_TYPE_NAME' não foi encontrado."
            exit 1
        fi

        CONFIG_PATH="$BASE_DIR/$current_model/$DATASET_NAME/$RUN_TYPE_NAME/synthetic/config.toml"
        if [ ! -f "$CONFIG_PATH" ]; then
            echo "Erro: O arquivo de configuração '$CONFIG_PATH' não foi encontrado."
            exit 1
        fi

        echo "conda run --no-capture-output -n SynthEnvironment python ./experiments_pipes/synthetic_generator_pipe.py --config $CONFIG_PATH"
        conda run --no-capture-output -n SynthEnvironment python ./experiments_pipes/synthetic_generator_pipe.py --config "$CONFIG_PATH"
    done
done

echo "Todos os testes foram concluídos com sucesso!"
