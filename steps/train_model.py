# steps/train_model_step.py
import os
import json
import math
import logging
from pathlib import Path
from typing import Dict, Any

import mlflow
# import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zenml import step

from utils.model_builders import build_custom_cnn, build_vgg16_tl, build_resnet50v2_tl


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    logger = logging.getLogger(f"train_step_{out_dir.name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    logger.info("Logger initialized at %s", str(log_file))
    return logger


def get_generators(train_dir: str, test_dir: str, model_key: str, batch_size: int, seed: int, val_split: float = 0.2):
    """
    Returns (train_gen, val_gen, test_gen).
    For TL models we use RGB 224x224; for custom CNN: grayscale 48x48 with validation split.
    """
    if model_key.startswith("custom_cnn"):
        datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=val_split)
        train_gen = datagen.flow_from_directory(
            train_dir, target_size=(48, 48), color_mode='grayscale',
            class_mode='categorical', batch_size=batch_size, subset='training', seed=seed
        )
        val_gen = datagen.flow_from_directory(
            train_dir, target_size=(48, 48), color_mode='grayscale',
            class_mode='categorical', batch_size=batch_size, subset='validation', seed=seed
        )
        test_gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
            test_dir, target_size=(48, 48), color_mode='grayscale',
            class_mode='categorical', batch_size=batch_size, shuffle=False
        )
        return train_gen, val_gen, test_gen
    else:
        # Transfer learning flows (RGB, 224x224). We don't create a val split here by default.
        datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                     rotation_range=10, zoom_range=0.1,
                                     width_shift_range=0.1, height_shift_range=0.1,
                                     horizontal_flip=True)
        train_gen = datagen.flow_from_directory(
            train_dir, target_size=(224, 224), color_mode='rgb',
            class_mode='categorical', batch_size=batch_size, shuffle=True
        )
        test_gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
            test_dir, target_size=(224, 224), color_mode='rgb',
            class_mode='categorical', batch_size=batch_size, shuffle=False
        )
        # No val_gen by default for TL; use test_gen as temporary val if needed
        return train_gen, None, test_gen


@step
def train_model_step(config: Dict[str, Any], model_key: str = "custom_cnn") -> Dict[str, Any]:
    """
    ZenML step that trains a model (custom CNN or TL) and logs artifacts to MLflow.
    Returns a summary dict: model name, artifact dir, metrics.
    """
    # Resolve directories
    project_dir = Path(config["project_dir"])
    train_dir = Path(config["train_dir"])
    test_dir = Path(config["test_dir"])
    artifact_dir = Path(config["artifact_dir"]) / model_key
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = setup_logger(artifact_dir)
    logger.info("Train dir: %s, Test dir: %s", train_dir, test_dir)

    # Hyperparams
    batch_size = int(config.get("batch_size", 64))
    epochs = int(config.get("epochs", 10))
    seed = int(config.get("seed", 42))
    num_classes = int(config.get("num_classes", 7))

    # Data
    try:
        train_gen, val_gen, test_gen = get_generators(str(train_dir), str(test_dir), model_key, batch_size, seed)
    except Exception as e:
        logger.exception("Failed to create generators: %s", e)
        raise

    # Model selection
    if model_key == "custom_cnn" or model_key == "custom_cnn_aug":
        model = build_custom_cnn(input_shape=(48, 48, 1), num_classes=num_classes)
    elif model_key == "vgg16_tl":
        model = build_vgg16_tl(input_shape=(224, 224, 3), num_classes=num_classes)
    elif model_key == "resnet50v2_tl":
        model = build_resnet50v2_tl(input_shape=(224, 224, 3), num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_key: {model_key}")

    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    logger.info("Model compiled: %s", model_key)

    # MLflow experiment context
    mlflow.set_experiment(config.get("mlflow_experiment", "emotion_detection"))
    # mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name=model_key):
        # Callbacks
        checkpoint_path = artifact_dir / f"{model_key}_best.keras"
        callbacks = [
            ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor="val_loss", mode="min", verbose=1),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1),
            CSVLogger(str(artifact_dir / "training.csv"))
        ]

        steps_per_epoch = max(1, math.ceil(train_gen.samples / batch_size))
        val_steps = None
        if val_gen:
            val_steps = max(1, math.ceil(val_gen.samples / batch_size))
        test_steps = max(1, math.ceil(test_gen.samples / batch_size))

        logger.info("Starting training: steps_per_epoch=%s val_steps=%s epochs=%d", steps_per_epoch, val_steps, epochs)
        try:
            history = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen if val_gen else test_gen,
                validation_steps=val_steps if val_steps else test_steps,
                epochs=epochs,
                callbacks=callbacks
            )
        except tf.errors.ResourceExhaustedError as e:
            logger.exception("Resource exhausted (OOM). Consider lowering batch_size or model size: %s", e)
            raise
        except Exception as e:
            logger.exception("Training failed unexpectedly: %s", e)
            raise

        # Save history
        history_path = artifact_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

        # Evaluate on test set
        logger.info("Running predictions on test set...")
        preds = model.predict(test_gen, steps=test_steps, verbose=1)
        pred_labels = preds.argmax(axis=1)
        true_labels = test_gen.classes
        class_names = list(test_gen.class_indices.keys())

        cr = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(true_labels, pred_labels)

        # Save metrics & artifacts
        (artifact_dir / "classification_report.json").write_text(json.dumps(cr, indent=2))
        np.save(artifact_dir / "confusion_matrix.npy", cm)

        # log to mlflow (some metrics autologged; log summary)
        test_acc = float(cr.get("accuracy", 0.0))
        macro_f1 = float(cr.get("macro avg", {}).get("f1-score", 0.0))
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("macro_f1", macro_f1)

        # Log small artifacts explicitly
        mlflow.log_artifact(str(artifact_dir / "classification_report.json"))
        mlflow.log_artifact(str(artifact_dir / "confusion_matrix.npy"))
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(checkpoint_path))

    # Return summary for pipeline
    summary = {
        "model_key": model_key,
        "artifact_dir": str(artifact_dir),
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "class_indices": test_gen.class_indices
    }
    logger.info("Training step completed for %s. Summary: %s", model_key, summary)
    return summary
