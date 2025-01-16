from pathlib import Path

from src.constantsconfigs.config import YoloSettingsUpdate
from ultralytics import settings
from src.constantsconfigs.constants import PROJECT_DEFAULT_PATH


def settings_update(cfg: YoloSettingsUpdate):
    """
    Update setting config before start.
    """
    settings.update(
        {
            "clearml": cfg.clearml,
            "mlflow": cfg.mlflow,
            "datasets_dir": PROJECT_DEFAULT_PATH,
        },
    )
