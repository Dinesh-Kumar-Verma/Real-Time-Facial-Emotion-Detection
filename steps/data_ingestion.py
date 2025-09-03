from utils.data_loader import load_data, unzip_folder
from pathlib import Path
from zenml import step

@step
def run_data_ingestion(
    source: str = "kaggle",
    destination: str = "./data",
    kaggle_dataset: str | None = "msambare/fer2013",
    gdrive_url: str | None = None,
) -> Path:
    """
    ZenML step for data ingestion from either Kaggle or Google Drive.

    Args:
        source (str): Data source ("kaggle" or "gdrive").
        destination (str): Local folder path.
        kaggle_dataset (str | None): Kaggle dataset name.
        gdrive_url (str | None): Google Drive link.

    Returns:
        Path: Path to dataset (downloaded or extracted).
    """
    dataset_path = load_data(source, destination, kaggle_dataset, gdrive_url)

    # Auto-unzip if zip file downloaded
    if dataset_path.suffix == ".zip":
        return unzip_folder(dataset_path, destination)
    return dataset_path


from pathlib import Path
from zenml import step
from utils.data_loader import load_data, unzip_folder
from utils.config_loader import load_config


@step
def run_data_ingestion() -> Path:
    """
    ZenML step for data ingestion from either Kaggle or Google Drive using parameters from config YAML.
    
    Args:
    source (str): Data source ("kaggle" or "gdrive").
    destination (str): Local folder path.
    kaggle_dataset (str | None): Kaggle dataset name.
    gdrive_url (str | None): Google Drive link.

    Returns:
        Path: Path to dataset (downloaded or extracted).
    """
    try:
        config = load_config("configs/data_config.yaml")
        dataset_cfg = config["dataset"]

        source = dataset_cfg["source"]  # You can also add source in YAML
        destination = dataset_cfg["data_path"]
        kaggle_dataset = dataset_cfg["kaggle_dataset"]
        gdrive_url = dataset_cfg["gdrive_url"]

        dataset_path = load_data(source, destination, kaggle_dataset, gdrive_url)

        if dataset_path.suffix == ".zip":
            return unzip_folder(dataset_path, destination)

        return dataset_path

    except Exception as e:
        raise RuntimeError(f"Data ingestion failed: {e}")
