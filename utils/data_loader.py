from utils.logger import logger
from zenml import step
import subprocess
import sys
from pathlib import Path
import gdown
import zipfile


def load_data(
    source: str,
    destination: str,
    kaggle_dataset: str | None = None,
    gdrive_url: str | None = None,
) -> Path:
    """
    Load data either from Google Drive or Kaggle.

    Args:
        source (str): "gdrive" or "kaggle".
        destination (str): Local path to save the data.
        kaggle_dataset (str | None): Kaggle dataset in format "owner/dataset".
        gdrive_url (str | None): Google Drive file shareable link.

    Returns:
        Path: Path to the downloaded file/folder.
    """
    try:
        dest_path = Path(destination).expanduser().resolve()
        dest_path.mkdir(parents=True, exist_ok=True)

        if source == "gdrive":
            if not gdrive_url:
                raise ValueError("Google Drive URL must be provided when source='gdrive'")
            logger.info("Downloading dataset from Google Drive...")
            output_path = dest_path / "data.zip"
            gdown.download(gdrive_url, str(output_path), quiet=False)
            logger.info(f"Dataset downloaded successfully: {output_path}")
            return output_path

        elif source == "kaggle":
            if not kaggle_dataset:
                raise ValueError("Kaggle dataset name must be provided when source='kaggle'")
            logger.info(f"Downloading dataset from Kaggle: {kaggle_dataset}...")
            cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", str(dest_path)]
            subprocess.run(cmd, check=True)
            logger.info(f"Dataset downloaded successfully into: {dest_path}")
            return dest_path

        else:
            raise ValueError("Source must be either 'gdrive' or 'kaggle'")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def unzip_folder(zip_path: str | Path, extract_to: str | Path) -> Path:
    """
    Unzip a folder safely with logging and error handling.

    Args:
        zip_path (str | Path): Path to the zip file.
        extract_to (str | Path): Directory where contents will be extracted.

    Returns:
        Path: Path to extracted files.
    """
    try:
        zip_file = Path(zip_path).expanduser().resolve()
        extract_path = Path(extract_to).expanduser().resolve()

        if not zip_file.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_file}")

        extract_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Unzipping {zip_file} into {extract_path}...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            # Safe extraction (prevents zip slip)
            for member in zip_ref.namelist():
                member_path = extract_path / member
                if not str(member_path.resolve()).startswith(str(extract_path)):
                    raise Exception(f"Unsafe path detected in zip: {member}")
            zip_ref.extractall(extract_path)

        logger.info(f"Files extracted successfully into: {extract_path}")
        return extract_path

    except zipfile.BadZipFile:
        logger.error(f"Bad zip file or corrupted: {zip_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to unzip file: {e}")
        raise