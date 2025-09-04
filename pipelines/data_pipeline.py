from zenml import pipeline
from steps.data_ingestion import run_data_ingestion
from steps.data_preprocessing import clean_image_dataset
from utils.config_loader import load_config

@pipeline
def data_pipeline():
    """Pipeline for ingesting dataset."""
    config = load_config("configs/data_config.yaml")
    dataset_cfg = config["dataset"]

    dataset_path = run_data_ingestion()
    summary = clean_image_dataset(data_dir=dataset_cfg["train_data_path"])
    return dataset_path, summary
