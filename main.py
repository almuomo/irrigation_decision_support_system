from datetime import date, timedelta
from time import perf_counter
import time

import pandas as pd

from scripts.get_token import get_siar_token
from scripts.etl_info_siar import run_info_pipeline
from scripts.etl_datos_siar import run_datos_pipeline, log_line
from scripts.etl_openmeteo import run_openmeteo_pipeline
from scripts.common.settings import DIM_ESTACION_PATH

DIM_EST_PATH = DIM_ESTACION_PATH


def load_stations_and_bajas() -> tuple[list[str], dict[str, str | None]]:
    """
    Carga desde dim_estacion la lista de estaciones activas y sus fechas de baja.
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
    """
    total_seconds = int(round(seconds))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    """
    Orquesta la ejecución principal del proceso ETL de datos SIAR.
    """
    t0 = perf_counter()
    log_line("[INFO] Inicio de ejecución main.py")

    try:
        # 1) Obtener token
        token = get_siar_token()
        log_line("[INFO] Token obtenido correctamente.")

        # 2) Pipeline INFO
        # Recomendable si quieres asegurarte de que dim_estacion esté actualizado
        log_line("[INFO] Ejecutando pipeline INFO...")
        run_info_pipeline(token=token)

        # log_line("[INFO] Esperando 65s tras INFO para evitar límites por minuto...")
        # time.sleep(65)

        # 3) Cargar estaciones y bajas
        estaciones, bajas = load_stations_and_bajas()
        log_line(f"[INFO] Número de estaciones cargadas: {len(estaciones)}")

        # 4) Pipeline DATOS
        log_line("[INFO] Ejecutando pipeline DATOS...")
        run_datos_pipeline(
            token=token,
            estaciones=estaciones,
            station_bajas=bajas,
            start_date="1999-01-01",
            end_date=str(date.today() - timedelta(days=1)),
            datos_calculados=True,
            min_access_buffer=5,
            min_records_buffer=2000,
            sleep_s=1.10,
            rebuild_history=False,
        )

        # 5) Pipeline OpenMeteo
        log_line("[INFO] Ejecutando pipeline OPEN-METEO...")
        run_openmeteo_pipeline(
            forecast_days=16,
            past_days=0,
            timezone_str="Europe/Madrid",
            include_closed=False,
            sleep_s=1.10,
            timeout=60,
            max_retries=5,
            datos_calculados_siar=True,
            overwrite_siar_gold_fact=True,
        )

        elapsed = perf_counter() - t0
        log_line(f"[INFO] Proceso finalizado correctamente. Duración total: {format_seconds(elapsed)}")

    except Exception as e:
        elapsed = perf_counter() - t0
        log_line(f"[ERROR] Fallo en main.py tras {format_seconds(elapsed)} -> {repr(e)}")
        raise


if __name__ == "__main__":
    main()