from zenml import pipeline
from steps.data_ingestion import run_data_ingestion
from utils.config_loader import load_config

@pipeline
def data_pipeline():
    """Pipeline for ingesting dataset."""
    dataset_path = run_data_ingestion()
    return dataset_path
