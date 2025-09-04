import os
import cv2
import imghdr
from pathlib import Path
from typing import Dict
from zenml import step
from utils.logger import logger

@step
def clean_image_dataset(data_dir: str) -> Dict:
    """
    ZenML step to clean image dataset in an OS-independent way:
      - removes files with unsupported extensions
      - validates readability with OpenCV
      - logs issues
    Returns a summary dictionary.
    """
    image_exts = ['jpeg', 'jpg', 'png']
    valid_exts = [ext.lower() for ext in image_exts]

    # Resolve dataset path (cross-platform)
    data_path = Path(data_dir).resolve()

    removed, processed = [], []

    for file_path in data_path.rglob("*"):  # recursively iterate
        if file_path.is_file():
            try:
                # Validate extension type (case-insensitive)
                file_type = (imghdr.what(file_path) or "").lower()

                if file_type not in valid_exts:
                    file_path.unlink()  # safer than os.remove
                    removed.append(str(file_path))
                    logger.warning(f"Removed unsupported: {file_path}")
                    continue

                # Try loading the image with OpenCV
                img = cv2.imread(str(file_path))
                if img is None:
                    file_path.unlink()
                    removed.append(str(file_path))
                    logger.error(f"Removed corrupted: {file_path}")
                else:
                    processed.append(str(file_path))

            except Exception as e:
                logger.exception(f"Issue with {file_path}: {e}")
                try:
                    file_path.unlink()
                    removed.append(str(file_path))
                except Exception as rm_err:
                    logger.error(f"Failed to remove {file_path}: {rm_err}")

    summary = {
        "data_dir": str(data_path),
        "removed_count": len(removed),
        "processed_count": len(processed),
        "removed_files": removed[:5], 
    }

    logger.info(f"Cleaning summary: {summary}")
    return summary
