from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
import time

import pandas as pd

from scripts.get_token import get_siar_token
from scripts.etl_info_siar import run_info_pipeline
from scripts.etl_datos_siar import run_datos_pipeline, log_line
from scripts.common.settings import DIM_ESTACION_PATH

DIM_EST_PATH = DIM_ESTACION_PATH

def load_stations_and_bajas() -> tuple[list[str], dict[str, str | None]]:
    """
    Carga desde dim_estacion la lista de estaciones activas y sus fechas de baja.

    Lee el fichero `data/gold/dim_estacion.parquet`, extrae la columna
    `estacion_codigo` y, si existe, la columna `fecha_baja`.

    Returns:
        tuple[list[str], dict[str, str | None]]:
            - Lista de códigos de estación normalizados en mayúsculas.
            - Diccionario con el formato estacion_codigo -> fecha_baja ISO
              (`YYYY-MM-DD`) o `None` si la estación no tiene fecha de baja.

    Raises:
        FileNotFoundError: Si no existe el fichero `dim_estacion.parquet`.
        ValueError: Si el fichero no contiene la columna `estacion_codigo`.
    """
    if not DIM_EST_PATH.exists():
        raise FileNotFoundError(f"No existe {DIM_EST_PATH}. Ejecuta primero run_info_pipeline.")

    df = pd.read_parquet(DIM_EST_PATH)

    if "estacion_codigo" not in df.columns:
        raise ValueError("dim_estacion.parquet no contiene 'estacion_codigo'.")

    fecha_baja_col = "fecha_baja" if "fecha_baja" in df.columns else None

    df["estacion_codigo"] = df["estacion_codigo"].astype(str).str.strip().str.upper()
    estaciones = df["estacion_codigo"].dropna().unique().tolist()

    bajas: dict[str, str | None] = {}
    if fecha_baja_col:
        fb = pd.to_datetime(df[fecha_baja_col], errors="coerce", utc=True)
        for cod, dt in zip(df["estacion_codigo"], fb):
            bajas[cod] = None if pd.isna(dt) else dt.date().isoformat()
    else:
        for cod in estaciones:
            bajas[cod] = None

    return estaciones, bajas


def format_seconds(seconds: float) -> str:
    """
    Convierte una duración en segundos a formato HH:MM:SS.

    Args:
        seconds (float): Duración en segundos.

    Returns:
        str: Duración formateada como horas, minutos y segundos.
    """
    total_seconds = int(round(seconds))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    """
    Orquesta la ejecución principal del proceso ETL de datos SIAR.

    Flujo principal:
    1. Registra el inicio de ejecución.
    2. Obtiene el token de acceso a la API SIAR.
    3. Opcionalmente puede ejecutar el pipeline de INFO.
    4. Carga las estaciones y fechas de baja desde `dim_estacion.parquet`.
    5. Ejecuta el pipeline incremental de datos diarios por estación.
    6. Registra en el log la duración total del proceso.
    7. En caso de error, registra la duración transcurrida y relanza la excepción.

    Returns:
        None
    """
    t0 = perf_counter()
    log_line("[INFO] Inicio de ejecución main.py")

    try:
        # 1️⃣ Obtener token
        token = get_siar_token()
        log_line("[INFO] Token obtenido correctamente.")

        # 2️⃣ Pipeline INFO
        log_line("[INFO] Ejecutando pipeline INFO...")
        run_info_pipeline(token=token)

        # log_line("[INFO] Esperando 65s tras INFO para evitar límites por minuto...")
        # time.sleep(65)

        # # 3️⃣ Pipeline DATOS
        # log_line("[INFO] Ejecutando pipeline DATOS...")
        
        # estaciones, bajas = load_stations_and_bajas()
        # log_line(f"[INFO] Número de estaciones cargadas: {len(estaciones)}")

        # run_datos_pipeline(
        #     token=token,
        #     estaciones=estaciones,
        #     station_bajas=bajas,
        #     start_date="2025-02-19",
        #     end_date=str(date.today() - timedelta(days=1)),
        #     datos_calculados=True,
        #     min_access_buffer=5,
        #     min_records_buffer=2000,
        #     sleep_s=14,
        # )

        # elapsed = perf_counter() - t0
        # log_line(f"[INFO] Proceso finalizado correctamente. Duración total: {format_seconds(elapsed)}")

    except Exception as e:
        elapsed = perf_counter() - t0
        log_line(f"[ERROR] Fallo en main.py tras {format_seconds(elapsed)} -> {repr(e)}")
        raise


if __name__ == "__main__":
    main()