import os
import re
from pathlib import Path
from typing import Callable, Dict

import mlflow
import pandas as pd
import yaml

from src.constantsconfigs.constants import PROJECT_DEFAULT_PATH
from src.constantsconfigs.config import YOLOTrainerConfig
from src.mlflow_tracking.cust_mlflow import (
    MLflowTracking,
    load_dataset_description,
)
from src.settings_update.yolo_settings_update import settings_update
from ultralytics import YOLO


class YOLOTrainer:  # noqa: WPS230
    """
    Main trainer.

    """

    def __init__(self, config: YOLOTrainerConfig, callbacks: Dict[str, Callable] = None):
        """
        Initialize the YOLOTrainer.
        :param config: Config fot train/inference mode.
        :param callbacks: List of callbacks
        """
        self.config_trainer = config
        self.cfg_file_yolo = self._load_model_config(config.cfg_model_path)
        if config.pretrained_path:
            self.cfg_file_yolo["training_params"]["model"] = PROJECT_DEFAULT_PATH / config.pretrained_path

        settings_update(config.yolo_settings_update)
        self.model = YOLO(self.cfg_file_yolo["training_params"]["model"])
        self.callbacks = callbacks
        self.mlflow = config.cfg_mlflow.mlflow_tracking_uri

        if self.mlflow:
            self.mlflow_tracking = MLflowTracking(
                self.mlflow,
                config.experiment_name,
            )
        self.save_nadir_results_path = None
        if config.path_save_res_nadirs:
            self.save_nadir_results_path = os.path.join(
                PROJECT_DEFAULT_PATH,
                config.path_save_res_nadirs,
                "results.csv",
            )

    def run_training(self):
        """
        Run the YOLO model training process.

        """
        if self.callbacks:
            self._set_callbacks()
        self._train_yolo_model(self.cfg_file_yolo)

    @classmethod
    def _load_model_config(cls, model_cfg_file: Path):
        """
        Load the YOLO model configuration from a YAML file.

        :param model_cfg_file: Path to the YOLO model configuration file.
        :return dict: Loaded model configuration as a dictionary.
        """
        with open(model_cfg_file, "r") as file_cfg:
            settings = yaml.safe_load(file_cfg)
            data = settings.get("data")
            if data and not Path(data).is_absolute():
                settings["data"] = PROJECT_DEFAULT_PATH / data
            return settings

    def _val_metrics_nadir(self):
        """
        Val SN4 model for each nadir.

        """
        list_data = sorted(
            os.listdir(
                PROJECT_DEFAULT_PATH / "configs/traindataconfigs/DataConfigsSN4Nadirs",
            ),
        )

        data_results = pd.DataFrame(
            columns=["nadir"] + list(self.model.trainer.metrics.keys()),
        )
        for nadir_cfg in list_data:
            results_model = self.model.val(
                data=(
                    PROJECT_DEFAULT_PATH
                    / Path(
                        "configs/traindataconfigs/DataConfigsSN4Nadirs",
                    )  # noqa: W503
                    / nadir_cfg  # noqa: W503
                ),
                split="test",
            )
            results_metrics = {name: round(results_model[name], 4) for name in results_model}
            results_metrics["nadir"] = re.findall(r"\d+", nadir_cfg).pop()
            data_results = pd.concat(
                [data_results, pd.DataFrame([results_metrics])],
                ignore_index=True,
            )

        data_results.to_csv(self.save_nadir_results_path)

        self.mlflow_tracking.log_custom_artifact(
            self.save_nadir_results_path,
            "ResultsNadir",
        )

    def _train_yolo_model(self, model_config: dict):
        """
        Train a YOLO model based on the provided configuration.

        :param model_config: YOLO model configuration as a dictionary.
        """
        if self.mlflow:
            with mlflow.start_run(
                run_name=self.config_trainer.experiment_name,
                description=load_dataset_description(
                    file_path=PROJECT_DEFAULT_PATH / self.config_trainer.cfg_data,
                ),
            ):
                self._do_train(model_config)
                return

        self._do_train(model_config)

    def _do_train(self, model_config: dict):
        """
        Train a YOLO model.

        """
        model_path = Path(model_config["training_params"]["data"])
        if not model_path.is_absolute():
            model_path = PROJECT_DEFAULT_PATH / model_path
        if model_path.exists() and model_path.is_dir() and any(model_path.iterdir()):
            msg = f"Отсутствуют данные модели в {model_path}"
            raise FileNotFoundError(msg)

        self.model.train(**model_config["training_params"], classes=self.config_trainer.need_classes)

        if self.save_nadir_results_path:
            self._val_metrics_nadir()

    def _set_callbacks(self):
        """
        Set callbacks.
        """
        for name_callback, func in self.callbacks.items():
            self.model.add_callback(name_callback, func)
