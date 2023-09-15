import os
import sys
import json
import asyncio
import mlflow
import logging
import click
import time
import tempfile
import yaml
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
import rasa
from rasa.model_training import train_nlu
from rasa.model_testing import test_nlu

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_domain(yaml_file):
    with open(yaml_file, 'r') as file:
        yaml_content = yaml.safe_load(file)

    json_content = {"num_" + key: len(value) for key, value in yaml_content.items() if isinstance(value, (list, dict))}
    # json_content = {"num_" + key: len(value) for key, value in yaml_content.items() if isinstance(value, list)}
    # json_content = {key: len(value) for key, value in yaml_content.items() if isinstance(value, list)}
    mlflow.log_params(json_content)
    return json.dumps(json_content)

def count_entity_errors():
    json_content = []
    try:
        with open("results/DIETClassifier_errors.json", 'r') as file:
            json_content = json.load(file)
    except:
        logger.info("DIETClassifier_errors.json not found")
    try:
        with open("results/CRFEntityExtractor_errors.json", 'r') as file:
            json_content = json.load(file)
    except:
        logger.info("CRFEntityExtractor_errors.json not found")

    # If the JSON content is a dictionary, count the number of key-value pairs
    if isinstance(json_content, dict):
        num_objects = len(json_content)
    # If the JSON content is a list, count the number of elements in the list
    elif isinstance(json_content, list):
        num_objects = len(json_content)
    else:
        num_objects = 0
    logger.info(f"entity_errors: {num_objects}")
    mlflow.log_metric("entity_errors", num_objects)

    return num_objects

def log_config_params(config_file):
    params = {"rasa_version": rasa.__version__, "intent_classifier": "", "entity_extrators": "", "featurizers": "", "llm_type": "", "llm_classifier_model": "", "llm_embeddings_type": "", "llm_embedding_model": ""}
    with open(config_file, "r") as file:
        yaml_content = yaml.safe_load(file)
    pipeline = yaml_content.get("pipeline", [])
    for component in pipeline:
        logger.debug(f"Component {component['name']}")
        # which intent classifier
        if (
            component["name"] == "DIETClassifier"
            or component["name"] == "MitieIntentClassifier"
            or component["name"] == "LogisticRegressionClassifier"
            or component["name"] == "SklearnIntentClassifier"
            or component["name"] == "KeywordIntentClassifier"
            or component["name"] == "rasa_plus.ml.LLMIntentClassifier"
        ):
            if len(params["intent_classifier"]):
                params["intent_classifier"] += ", " + component["name"]
            else:
                params["intent_classifier"] += component["name"]
            # For LLM"s get model_name
            if component["name"] == "rasa_plus.ml.LLMIntentClassifier":
                params["llm_type"] = component["llm"].get("type", "openai")
                # Set llm_classifier_model
                if component["llm"].get("model_name") is None and params["llm_type"] == "openai":
                    params["llm_classifier_model"] = "text-davinici-003"
                elif params["llm_type"] == "openai":
                    params["llm_classifier_model"] = component["llm"].get("model_name")
                elif params["llm_type"] == "cohere":
                    params["llm_classifier_model"] = component["llm"].get("model")
                elif params["llm_type"] == "vertexai":
                    params["llm_classifier_model"] = component["llm"].get("model_name")
                elif params["llm_type"] == "huggingface_hub":
                    params["llm_classifier_model"] = component["llm"].get("repo_id")
                elif params["llm_type"] == "llamacpp":
                    params["llm_classifier_model"] = os.path.basename(component["embeddings"].get("model_path"))
                # Set llm_embedding_model
                if component.get("embeddings") == None:
                    params["llm_embeddings_type"] = "openai"
                else:
                    params["llm_embeddings_type"] = component["embeddings"].get("type", "openai")
                    if component["embeddings"].get("model_name") is None and params["llm_embeddings_type"] == "openai":
                        params["llm_embedding_model"] = "text-embedding-ada-002"
                    elif params["llm_embeddings_type"] == "openai":
                        params["llm_embedding_model"] = component["embeddings"].get("model")
                    elif params["llm_embeddings_type"] == "cohere":
                        params["llm_embedding_model"] = component["embeddings"].get("model")
                    elif params["llm_embeddings_type"] == "spacy":
                        params["llm_embedding_model"] = component["embeddings"].get("model")
                    elif params["llm_embeddings_type"] == "vertexai":
                        params["llm_embedding_model"] = component["embeddings"].get("model_name")
                    elif params["llm_embeddings_type"] == "huggingface_hub":
                        params["llm_embedding_model"] = component["embeddings"].get("remo_id")
                    elif params["llm_embeddings_type"] == "huggingface_instruct":
                        params["llm_embedding_model"] = component["embeddings"].get("model_name")
                    elif params["llm_embeddings_type"] == "llamacpp":
                        params["llm_embedding_model"] = os.path.basename(component["embeddings"].get("model_path"))
        # which featurizers
        if (
            component["name"] == "MitieFeaturizer"
            or component["name"] == "SpacyFeaturizer"
            or component["name"] == "ConveRTFeaturizer"
            or component["name"] == "LanguageModelFeaturizer"
            or component["name"] == "RegexFeaturizer"
            or component["name"] == "CountVectorsFeaturizer"
            or component["name"] == "LexicalSyntacticFeaturizer"
        ):
            if len(params["featurizers"]):
                params["featurizers"] += ", " + component["name"]
            else:
                params["featurizers"] += component["name"]
        # which entity extrators
        if (
            component["name"] == "MitieEntityExtractor"
            or component["name"] == "SpacyEntityExtractor"
            or component["name"] == "CRFEntityExtractor"
            or component["name"] == "DucklingEntityExtractor"
            or component["name"] == "RegexEntityExtractor"
            or component["name"] == "EntitySynonymMapper"
        ):
            if len(params["entity_extrators"]):
                params["entity_extrators"] += ", " + component["name"]
            else:
                params["entity_extrators"] += component["name"]
        # check for DIET entity extractions
        if component["name"] == "DIETClassifier":
            if component.get("entity_recognition") is None or (component.get("entity_recognition") and component["entity_recognition"] is True):
                if len(params["entity_extrators"]):
                    params["entity_extrators"] += ", " + component["name"]
                else:
                    params["entity_extrators"] += component["name"]
    logger.debug(f"Params: {params}")
    mlflow.log_params(params)


def latest_model(model_dir):
    # Get a list of all files in the specified directory
    files = os.listdir(model_dir)

    # Filter out directories and get only regular files
    files = [f for f in files if os.path.isfile(os.path.join(model_dir, f))]

    # Sort the files by their last modification time in descending order
    sorted_files = sorted(
        files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True
    )

    # Return the name of the latest file (first element in the sorted list)
    if sorted_files:
        return sorted_files[0]
    else:
        return None  # Return None if the directory is empty


def _extract_intent_report(file):
    try:
        with open(file, "r") as f:
            metrics = json.load(f)
            json_object = {"accuracy": metrics["accuracy"], **metrics["weighted avg"]}
    except FileNotFoundError:
        logger.error(f"Unable to report metrics, no {file} found")
        json_object = {}

    return json_object


def extract_metric(file, metric_name):
    with open(file, "r") as f:
        metrics = json.load(f)

    return metrics[metric_name]


def process_results(results_dir):
    files = os.listdir(results_dir)
    if not files:
        logger.info("Test failed. No results have been generated")
        metrics = {
            "support": 0,
            "f1-score": 0,
            "f1-entity": 0,
            # "elapsed_time": time.time() - start_time,
        }
        mlflow.log_metrics(metrics=metrics)
        return metrics
    logging.info(f"Results directory: {files}")
    mlflow.log_artifacts("results", artifact_path="results")
    # extensions = ["json", "pdf", "png"]
    # for file in files:
    #     if os.path.splitext(file)[1][1:] in extensions:
    #         logging.debug(f"log_artifact({file}), type: {type(file)}")
    #         mlflow.log_artifact(file,  artifact_path="reports")
    intent_stats = _extract_intent_report(f"{results_dir}/intent_report.json")
    mlflow.log_metrics(metrics=intent_stats)
    intent_errors = 0
    f1_entity = 0
    try:
        with open(f"{results_dir}/intent_errors.json", "r") as file:
            data = json.load(file)
            intent_errors = len(data)
    except FileNotFoundError:
        logger.error(
            "Unable to count intent errors, no results/intent_errors.json found"
        )
    # DIETClassifier_report.json
    try:
        f1_entity = extract_metric(
            f"{results_dir}/DIETClassifier_report.json", "accuracy"
        )  # if it crashes, you probably didn't use entities, either add entities or remove this line
        logger.info(f"DIETClassifier_report accuracy f1-entity: {f1_entity}")
    except FileNotFoundError:
        logger.info(
            "Unable to log entity f1. You need to provide entities tagging within your examples"
        )
    # CRFEntityExtractor_report.json
    try:
        f1_entity = extract_metric(
            f"{results_dir}/CRFEntityExtractor_report.json", "accuracy"
        )  # if it crashes, you probably didn't use entities, either add entities or remove this line
        logger.info(f"CRFEntityExtractor_report accuracy f1-entity: {f1_entity}")
    except FileNotFoundError:
        logger.info(
            "Unable to log entity f1. You need to provide entities tagging within your examples"
        )
    metrics = {
        "f1-entity": f1_entity,
        "intent_errors": intent_errors,
        # "test_time": end_time - start_time,
    }
    mlflow.log_metrics(metrics=metrics)
    return metrics


@click.command(help="Read rasa results directory and pass to mlflow")
@click.option("--train-time")
@click.option("--test-time")
@click.option("--results", default="./results")
@click.option("--experiment-name")
@click.option("--run-name")
@click.option("--config", default="config.yml")
def workflow(experiment_name, results, run_name, train_time, config, test_time):
    logger.debug(f"train_time: {train_time}")
    mlflow.set_tracking_uri("http://localhost:5000")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    logger.debug("start_run")
    with mlflow.start_run():
        if run_name:
            mlflow.set_tag("mlflow.runName", run_name)
        if train_time:
            mlflow.log_metric("train_time", train_time)
        if test_time:
            mlflow.log_metric("test_time", test_time)
        log_config_params(config)
        model_file = latest_model("models")
        mlflow.log_artifact(f"./models/{model_file}", artifact_path="model")
        mlflow.log_artifact(f"./{config}", artifact_path="config")

        # model_uri = os.path.join(run.info.artifact_uri, "model")
        # model_path = _transform_uri_to_path(model_uri)
        process_results(results)
        process_domain("domain.yml")
        count_entity_errors()
        # test_model = _get_or_run("test", {"model_path": model_path, "test": test_data})
        logger.info("Testing complete")


if __name__ == "__main__":
    workflow()
