import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _CONFIG = json.load(f)

API_CONFIG = _CONFIG["api"]
PATHS_CONFIG = _CONFIG["paths"]


def get_path(key: str) -> Path:
    return (PROJECT_ROOT / PATHS_CONFIG[key]).resolve()


SIAR_BASE_URL = API_CONFIG["siar_base_url"]

DATA_BRONZE_INFO_DIR = get_path("data_bronze_info_dir")
DATA_BRONZE_DATOS_DIR = get_path("data_bronze_datos_dir")
DATA_SILVER_DIR = get_path("data_silver_dir")
DATA_GOLD_DIR = get_path("data_gold_dir")
DIM_ESTACION_PATH = get_path("dim_estacion_path")
CHECKPOINT_SIAR_DIARIOS_ESTACION = get_path("checkpoint_siar_diarios_estacion")
LOGS_DIR = get_path("logs_dir")