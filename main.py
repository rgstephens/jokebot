import os
import sys
import json
import asyncio
import mlflow
import logging
import click
import time
import yaml
import tempfile
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
import rasa
from rasa import model_training
# from rasa.model_training import train_nlu, train
from rasa.model_testing import test_nlu

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def _transform_uri_to_path(uri, sub_directory=""):
    """transform run returned artifacts uri to path

    Args:
        uri (uri): uri returned by a run
        sub_directory (str, optional): sub directory to select artifact from if necessary. Defaults to "".

    Returns:
        path: path of the artifact returned
    """
    parsed_url = urlparse(uri)
    # Get the path from the parsed URL
    path = parsed_url.path
    # Make the path absolute using os.path.abspath()
    abs_path = os.path.abspath(path)
    # Get the list of files in the directory
    if sub_directory:
        files_path = os.path.join(abs_path, sub_directory)
        files = os.listdir(files_path)
        # Get the first file in the list (assuming there is at least one file in the directory)
        file = files[0]
        # Create the new path by combining the file name with the current working directory
        return os.path.join(files_path, file)
    return abs_path


def _generate_config(template_path, params, destination_file):
    """generates a configuration file based on the given template and parameters.

    Args:
        template_path (str): path to template config
        params (dict): key:value pari dict for params
        destination (str): destination file
    Returns:
        file_path (str): path to generated configuration file
    """
    # logger.info(f"Writing config to {destination}")
    # file_path = os.path.join(destination)
    # file_path = os.path.join(destination, "config_mlflow_template.yml")
    if isinstance(params, (str)):
        params = json.loads(params)
    with open(template_path) as f:
        run_config_yml = f.read().format(**params)
        # destination_file.write(run_config_yml.encode("utf-8"))
        # destination_file.write(run_config_yml)
        # logger.debug(run_config_yml)
        with open(destination_file, "w+") as temp_f:
            temp_f.write(run_config_yml)
            logger.debug(run_config_yml)
    return destination_file


def process_domain(yaml_file):
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    json_content = {
        "num_" + key: len(value)
        for key, value in yaml_content.items()
        if isinstance(value, (list, dict))
    }
    # json_content = {"num_" + key: len(value) for key, value in yaml_content.items() if isinstance(value, list)}
    # json_content = {key: len(value) for key, value in yaml_content.items() if isinstance(value, list)}
    mlflow.log_params(json_content)
    return json.dumps(json_content)


def count_entity_errors():
    json_content = []
    try:
        with open("results/DIETClassifier_errors.json", "r") as file:
            json_content = json.load(file)
    except:
        logger.info("DIETClassifier_errors.json not found")
    try:
        with open("results/CRFEntityExtractor_errors.json", "r") as file:
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
    params = {
        "rasa_version": rasa.__version__,
        "intent_classifier": "",
        "entity_extrators": "",
        "featurizers": "",
        "llm_type": "",
        "llm_classifier_model": "",
        "llm_embeddings_type": "",
        "llm_embedding_model": "",
    }
    with open(config_file, "r") as file:
        yaml_content = yaml.safe_load(file)
    pipeline = yaml_content.get("pipeline", [])
    if pipeline:
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
                    if (
                        component["llm"].get("model_name") is None
                        and params["llm_type"] == "openai"
                    ):
                        params["llm_classifier_model"] = "text-davinici-003"
                    elif params["llm_type"] == "openai":
                        params["llm_classifier_model"] = component["llm"].get(
                            "model_name"
                        )
                    elif params["llm_type"] == "cohere":
                        params["llm_classifier_model"] = component["llm"].get(
                            "model"
                        )
                    elif params["llm_type"] == "vertexai":
                        params["llm_classifier_model"] = component["llm"].get(
                            "model_name"
                        )
                    elif params["llm_type"] == "huggingface_hub":
                        params["llm_classifier_model"] = component["llm"].get(
                            "repo_id"
                        )
                    elif params["llm_type"] == "llamacpp":
                        params["llm_classifier_model"] = os.path.basename(
                            component["llm"].get("model_path")
                        )
                    # Set llm_embedding_model
                    if component.get("embeddings") == None:
                        params["llm_embeddings_type"] = "openai"
                    else:
                        params["llm_embeddings_type"] = component[
                            "embeddings"
                        ].get("type", "openai")
                        if (
                            component["embeddings"].get("model_name") is None
                            and params["llm_embeddings_type"] == "openai"
                        ):
                            params[
                                "llm_embedding_model"
                            ] = "text-embedding-ada-002"
                        elif params["llm_embeddings_type"] == "openai":
                            params["llm_embedding_model"] = component[
                                "embeddings"
                            ].get("model")
                        elif params["llm_embeddings_type"] == "cohere":
                            params["llm_embedding_model"] = component[
                                "embeddings"
                            ].get("model")
                        elif params["llm_embeddings_type"] == "spacy":
                            params["llm_embedding_model"] = component[
                                "embeddings"
                            ].get("model")
                        elif params["llm_embeddings_type"] == "vertexai":
                            params["llm_embedding_model"] = component[
                                "embeddings"
                            ].get("model_name")
                        elif params["llm_embeddings_type"] == "huggingface_hub":
                            params["llm_embedding_model"] = component[
                                "embeddings"
                            ].get("remo_id")
                        elif (
                            params["llm_embeddings_type"] == "huggingface_instruct"
                        ):
                            params["llm_embedding_model"] = component[
                                "embeddings"
                            ].get("model_name")
                        elif params["llm_embeddings_type"] == "llamacpp":
                            params["llm_embedding_model"] = os.path.basename(
                                component["embeddings"].get("model_path")
                            )
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
                if component.get("entity_recognition") is None or (
                    component.get("entity_recognition")
                    and component["entity_recognition"] is True
                ):
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
        files,
        key=lambda x: os.path.getmtime(os.path.join(model_dir, x)),
        reverse=True,
    )

    # Return the name of the latest file (first element in the sorted list)
    if sorted_files:
        return sorted_files[0]
    else:
        return None  # Return None if the directory is empty


def train(config, domain, training_data, temp_model_dir, force_training):
    """
    Trains a rasa NLU model using the given configuration and training file path.
    It also logs in mlflow the training duration as a metric and the model as an artifact.

    Args:
        config (string): Path to the config file (config.yml) for NLU.
        training (string): Path to the NLU training data (training_data.yml).
    Returns:
        None
    """

    logger.info(f"Writing model to directory {temp_model_dir}")
    logger.info(f"Starting train, data: {training_data}")
    # with tempfile.TemporaryDirectory() as temp_model_dir:
    #     logger.info(f"Writing model to directory {temp_model_dir}")
    start_time = time.time()
    # https://rasa.com/docs/rasa/reference/rasa/model_training/
    model_training.train(
        domain=domain,
        config=config,
        training_files=[training_data],
        output=temp_model_dir,
        nlu_additional_arguments={},
    )
    # train_nlu(
    #     config=config,
    #     nlu_data=training_data,
    #     output=temp_model_dir,
    #     additional_arguments={},
    # )
    duration = time.time() - start_time
    logger.info(f"Train completed: {duration:.2f}s")
    mlflow.log_metric("train_time", duration)
    mlflow.log_artifact(temp_model_dir, artifact_path="model")
    mlflow.log_artifact(config, artifact_path="config")
    return temp_model_dir


def _extract_intent_report(file):
    with open(file, "r") as f:
        metrics = json.load(f)
        json_object = {
            "accuracy": metrics["accuracy"],
            **metrics["weighted avg"],
        }

    return json_object


def extract_metric(file, metric_name):
    with open(file, "r") as f:
        metrics = json.load(f)

    return metrics[metric_name]


def process_results(results_dir):
    files = os.listdir(results_dir)
    logging.info(f"Results directory: {files}")
    if not files:
        logger.error("Test failed. No results have been generated")
        metrics = {
            "support": 0,
            # "f1-score": 0,
            # "f1-entity": 0,
            # "elapsed_time": time.time() - start_time,
        }
        mlflow.log_metrics(metrics=metrics)
        return metrics
    logging.info(f"Results directory: {files}")
    mlflow.log_artifacts("results", artifact_path="results")
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
        logger.info(
            f"CRFEntityExtractor_report accuracy f1-entity: {f1_entity}"
        )
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


def evaluate_nlu_model(model_path, test):
    """
    Evaluates a trained Rasa NLU model on a set of NLU test data.
    Args:
    model_path: Path to the last trained model in mlrun files
    test: Path pointing to the test files
    """
    logger.info(f"Starting test, model: {model_path}")
    start_time = time.time()

    f1_entity = 0
    with tempfile.TemporaryDirectory() as temp_results_dir:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            test_nlu(  # faster than both compute_metrics, run_evaluation
                model=model_path,
                nlu_data=test,
                output_directory=temp_results_dir,
                additional_arguments={"errors": True},
            )
        )
        logger.info("Processing test results")
        process_results(temp_results_dir)
        count_entity_errors()
        logger.info("Completed processing test results")
        mlflow.log_artifacts(temp_results_dir, artifact_path="reports")

    end_time = time.time()
    mlflow.log_metric("test_time", end_time - start_time)
    logger.info(f"Testing complete: {end_time - start_time}")


@click.command(help="Perform unique run of the code with provided params.")
@click.option("--force-training", default=False)
@click.option("--ted-epochs", default="40")
@click.option("--diet-epochs", default="100")
@click.option("--spacy-model", default="en_core_web_md")
@click.option("--config", default="config.yml")
@click.option("--domain", default="domain.yml")
@click.option("--config-template")
@click.option("--results", default="./results")
@click.option("--train-data", default="train_test_split/training_data.yml")
@click.option("--test-data", default="train_test_split/test_data.yml")
@click.option("--experiment-name", default="rasa-financial-demo")
@click.option("--run-name")
def workflow(
    experiment_name,
    ted_epochs,
    diet_epochs,
    spacy_model,
    config,
    domain,
    config_template,
    results,
    train_data,
    test_data,
    run_name,
    force_training,
):
    logger.info(
        f"\033[94mSetting up mlflow experiment: {experiment_name}, run name: {run_name}\033[0m"
    )
    tracking_uri = mlflow.tracking.get_tracking_uri()
    if tracking_uri.startswith("file:"):
        tracking_uri = "http://localhost:5000/"
        mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Writing to mlflow server at: {tracking_uri}")
    with mlflow.start_run() as run:
        if run_name:
            logger.info(f"Setting mlflow runName: {run_name}")
            mlflow.set_tag("mlflow.runName", run_name)
        # if you specified a config_template on the command line, then we'll use it
        if config_template:
            config_file = f"{tempfile.NamedTemporaryFile().name}.yml"
            config_file = _generate_config(
                config_template, run.data.params, config_file
            )
        # if you specified a config on the command line, then we'll use that
        else:
            config_file = config
        log_config_params(config)
        process_domain("domain.yml")
        with tempfile.TemporaryDirectory() as temp_model_dir:
            train(config_file, domain, train_data, temp_model_dir, force_training)
            evaluate_nlu_model(temp_model_dir, test_data)


if __name__ == "__main__":
    workflow()
