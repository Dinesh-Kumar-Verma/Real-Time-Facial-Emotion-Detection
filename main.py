# from pipelines.data_pipeline import data_pipeline

# if __name__ == "__main__":
#     pipe = data_pipeline()

# run_pipeline.py
import argparse
from pathlib import Path
from zenml.client import Client
from zenml.pipelines import pipeline as zen_pipeline_decorator
from pipelines.training_pipeline import training_pipeline
from steps.train_model import train_model_step
from utils.config_loader import load_config

def run(config_path: str, model_key: str):
    config = load_config(config_path)
    # ZenML usage: run the step directly by calling the step function (ZenML will create an execution)
    # We'll call the @step as a function with config and model_key
    # Note: this assumes zenml is initialized (zenml init) and stack is configured if using orchestrators/artifact stores.
    result = train_model_step(config=config, model_key=model_key)
    print("Run result:", result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training pipeline (ZenML step with MLflow).")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, default="custom_cnn", help="Model key to train")
    args = parser.parse_args()
    run(args.config, args.model)

