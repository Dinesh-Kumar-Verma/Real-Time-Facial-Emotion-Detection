import yaml
from pathlib import Path
from utils.logger import logger

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load YAML config file into a dictionary.
    Includes error handling and logging.
    """
    config_file = Path(config_path).expanduser().resolve()

    try:
        logger.info(f"Loading configuration from: {config_file}")

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at {config_file}")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Config file is not properly structured as a dictionary")

        logger.info("Configuration loaded successfully")
        return config

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading config: {e}")
        raise


# --- Example usage ---
if __name__ == "__main__":
    try:
        cfg = load_config()
        logger.info(f"Raw dataset path: {cfg['dataset']['raw_path']}")
        logger.info(f"Image size: {cfg['dataset']['image_size']}")
        logger.info(f"Classes: {cfg['classes']}")
    except Exception as e:
        logger.critical(f"Failed to load config: {e}")
