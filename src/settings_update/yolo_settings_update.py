from src.constantsconfigs.config import YoloSettingsUpdate
from ultralytics import settings
from src.constantsconfigs.constants import PROJECT_DEFAULT_PATH
from src.utils.load_config import load_config


def settings_update(cfg: YoloSettingsUpdate):
    """
    Update setting config before start.
    """
    settings.update(
        {
            "clearml": cfg.clearml,
            "mlflow": cfg.mlflow,
            "datasets_dir": str(cfg.data_path),
        },
    )


def main():
    config_path = "configs/trainerconfigs/trainer_config_cars.yaml"
    config = load_config(PROJECT_DEFAULT_PATH / config_path)
    settings_update(config.yolo_settings_update)

if __name__ == '__main__':
    main()