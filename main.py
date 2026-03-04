from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from scripts.get_token import get_siar_token
from scripts.etl_info_siar import run_info_pipeline
from scripts.etl_datos_siar import run_incremental_diarios_por_estacion


DIM_EST_PATH = Path("data/gold/dim_estacion.parquet")

def load_stations_and_bajas() -> tuple[list[str], dict[str, str | None]]:
    """
    Devuelve:
      - lista de códigos de estación
      - dict estacion_codigo -> fecha_baja (YYYY-MM-DD) o None
    """
    if not DIM_EST_PATH.exists():
        raise FileNotFoundError(f"No existe {DIM_EST_PATH}. Ejecuta primero run_info_pipeline.")

    df = pd.read_parquet(DIM_EST_PATH)

    if "estacion_codigo" not in df.columns:
        raise ValueError("dim_estacion.parquet no contiene 'estacion_codigo'.")

    # fecha_baja puede no existir según tu pipeline de INFO
    fecha_baja_col = "fecha_baja" if "fecha_baja" in df.columns else None

    df["estacion_codigo"] = df["estacion_codigo"].astype(str).str.strip().str.upper()

    estaciones = df["estacion_codigo"].dropna().unique().tolist()

    bajas: dict[str, str | None] = {}
    if fecha_baja_col:
        # convertir a fecha ISO si viene como datetime
        fb = pd.to_datetime(df[fecha_baja_col], errors="coerce", utc=True)
        for cod, dt in zip(df["estacion_codigo"], fb):
            bajas[cod] = None if pd.isna(dt) else dt.date().isoformat()
    else:
        for cod in estaciones:
            bajas[cod] = None

    return estaciones, bajas


def main():

    # -------------------------------------------------
    # 1️⃣ Obtener token (con cache en memoria)
    # -------------------------------------------------
    token = get_siar_token()

    # -------------------------------------------------
    # 2️⃣ Pipeline INFO (CCAA, provincias, estaciones)
    # -------------------------------------------------
    # ⚠️ Ya lo tienes generado, por eso lo dejamos comentado.
    # Descomenta solo si quieres refrescar dimensiones.
    #
    # print("[INFO] Ejecutando pipeline INFO...")
    # run_info_pipeline(token=token)

    # -------------------------------------------------
    # 3️⃣ Pipeline DATOS (incremental diarios por estación)
    # -------------------------------------------------
    print("[INFO] Ejecutando pipeline DATOS (Diarios + ESTACION)...")

    estaciones, bajas = load_stations_and_bajas()

    run_incremental_diarios_por_estacion(
        token=token,
        estaciones=estaciones,
        station_bajas=bajas, 
        start_date="2025-02-19",
        end_date=str(date.today() - timedelta(days=1)),
        datos_calculados=True,   # empieza en False para reducir volumen
        min_access_buffer=5,
        min_records_buffer=2000,
        sleep_s=14,
    )

    print("[INFO] Proceso finalizado.")


if __name__ == "__main__":
    main()