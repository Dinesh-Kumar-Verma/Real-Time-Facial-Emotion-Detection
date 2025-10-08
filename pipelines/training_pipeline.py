# pipelines/training_pipeline.py
from zenml import pipeline
from steps.train_model import train_model_step


@pipeline
def training_pipeline():
    """ZenML pipeline orchestration."""
    # This defines the pipeline DAG. If you need multiple steps (ingest->clean->train->eval) expand here.
    train_model_step()