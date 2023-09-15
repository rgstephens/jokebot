#/bin/bash
if [ -n "$1" ]; then
    config="$1"
else
    # If the first parameter is not provided, set a default value
    config="config.yml"
fi
if [ -n "$2" ]; then
    export MLFLOW_EXPERIMENT_NAME="$2"
else
    # If the first parameter is not provided, set a default value
    export MLFLOW_EXPERIMENT_NAME=""
fi
export RASA_PRO_BETA_LLM_INTENT=true
export RASA_PRO_BETA_DOCSEARCH=true
export LLAMA_METAL=1
echo Starting script with config: ${config}, experiment name: ${MLFLOW_EXPERIMENT_NAME}
echo Running mlflow run . -P config=${config} -P experiment-name=${MLFLOW_EXPERIMENT_NAME} -P run-name=${config%%.*}
mlflow run . -P config=${config} -P experiment-name=${MLFLOW_EXPERIMENT_NAME} -P run-name=${config%%.*}
