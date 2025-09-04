from pathlib import Path
import yaml

# Load YAML configuration
with open("project_structure.yaml") as f:
    config = yaml.safe_load(f)

# base_path = Path(config["project_name"])
base_path = Path(".")

# Create directories
for d in config["directories"]:
    (base_path / d).mkdir(parents=True, exist_ok=True)

# Create files
for file in config["files"]:
    file_path = base_path / file
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dirs exist
    file_path.touch(exist_ok=True)  # Create empty file if not exists

print(f"Project structure '{config['project_name']}' created successfully.")
