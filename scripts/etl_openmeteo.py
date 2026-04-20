from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from scripts.common.settings import (
    DATA_BRONZE_DATOS_DIR,
    DATA_SILVER_DIR,
    DATA_GOLD_DIR,
    DIM_ESTACION_PATH,
    LOGS_DIR,
)

OPENMETEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# -------------------------
# PATHS
# -------------------------
BRONZE_DIR = DATA_BRONZE_DATOS_DIR / "openmeteo_diarios_estacion"
SILVER_PATH = DATA_SILVER_DIR / "openmeteo_diarios_estacion.parquet"
MERGED_CONTEXT_PATH = DATA_GOLD_DIR / "fact_siar_diarios_estacion_calc1_openmeteo_context.parquet"
OPENMETEO_CHECKPOINT_PATH = DATA_SILVER_DIR / "openmeteo_diarios_estacion_checkpoint.json"
OPENMETEO_INPROGRESS_SILVER_PATH = DATA_SILVER_DIR / "openmeteo_diarios_estacion_inprogress.parquet"

# Gold final: se sobrescribe el mismo fact de SIAR con el dataset unificado
DEFAULT_SIAR_GOLD_FACT_NAME = "fact_siar_diarios_estacion_calc1.parquet"

DEFAULT_TIMEOUT = 60

# -------------------------
# VARIABLES OPEN-METEO
# -------------------------
# Variables estándar confirmadas para /v1/forecast
OPENMETEO_DAILY_VARS_STANDARD = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_mean",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_mean",
    "apparent_temperature_min",
    "sunrise",
    "sunset",
    "daylight_duration",
    "sunshine_duration",
    "uv_index_max",
    "uv_index_clear_sky_max",
    "rain_sum",
    "showers_sum",
    "snowfall_sum",
    "precipitation_sum",
    "precipitation_hours",
    "precipitation_probability_max",
    "precipitation_probability_mean",
    "precipitation_probability_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

# Variables adicionales que tienen sentido para cuadrar mejor con SIAR
# y que siguen la nomenclatura oficial publicada por Open-Meteo
OPENMETEO_DAILY_VARS_ADDITIONAL_SAFE = [
    "cloud_cover_max",
    "cloud_cover_mean",
    "cloud_cover_min",
    "relative_humidity_2m_max",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_min",
    "dew_point_2m_max",
    "dew_point_2m_mean",
    "dew_point_2m_min",
    "pressure_msl_max",
    "pressure_msl_mean",
    "pressure_msl_min",
    "surface_pressure_max",
    "surface_pressure_mean",
    "surface_pressure_min",
    "wind_speed_10m_mean",
    "wind_speed_10m_min",
    "wind_gusts_10m_mean",
    "wind_gusts_10m_min",
    "cape_max",
    "cape_mean",
    "cape_min",
]

# Variables candidatas que pueden existir según modelo/doc adicional.
# Si alguna no la acepta /v1/forecast, el código la elimina automáticamente
# y vuelve a intentar la petición sin romper el pipeline.
OPENMETEO_DAILY_VARS_CANDIDATES = [
    "visibility_max",
    "visibility_mean",
    "visibility_min",
    "wet_bulb_temperature_2m_max",
    "wet_bulb_temperature_2m_mean",
    "wet_bulb_temperature_2m_min",
    "vapour_pressure_deficit_max",
    "leaf_wetness_probability_mean",
    "growing_degree_days_base_0_limit_50",
    "snowfall_water_equivalent_sum",
]

OPENMETEO_DAILY_VARS_REQUESTED = (
    OPENMETEO_DAILY_VARS_STANDARD
    + OPENMETEO_DAILY_VARS_ADDITIONAL_SAFE
    + OPENMETEO_DAILY_VARS_CANDIDATES
)

# -------------------------
# DICCIONARIO DE MAPEADO A CANÓNICO SIAR
# -------------------------
# Las columnas de la izquierda son los nombres canónicos del fact SIAR.
# Las de la derecha son las columnas Open-Meteo que se usarán para rellenarlas
# cuando la fuente sea predictiva.
OPENMETEO_TO_SIAR_CANONICAL_MAP: Dict[str, str] = {
    "TempMedia": "temperature_2m_mean",
    "TempMax": "temperature_2m_max",
    "TempMin": "temperature_2m_min",
    "HumedadMedia": "relative_humidity_2m_mean",
    "HumedadMax": "relative_humidity_2m_max",
    "humedadMin": "relative_humidity_2m_min",
    "VelViento": "wind_speed_10m_mean",
    "DirViento": "wind_direction_10m_dominant",
    "VelVientoMax": "wind_speed_10m_max",
    "Radiacion": "shortwave_radiation_sum",
    "Precipitacion": "precipitation_sum",
    "EtPMon": "et0_fao_evapotranspiration",
}

UNIFIED_META_COLS = [
    "fuente_dato",
    "tipo_dato",
    "forecast_run_utc",
    "forecast_reference_date",
    "horizonte_dias",
    "om_daily_units_json",
    "om_requested_daily_vars_json",
    "om_response_latitude",
    "om_response_longitude",
    "om_response_elevation",
    "om_utc_offset_seconds",
    "om_timezone",
    "om_timezone_abbreviation",
    "om_generationtime_ms",
]


# -------------------------
# HELPERS
# -------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_line(msg: str) -> None:
    ensure_dir(LOGS_DIR)
    log_path = LOGS_DIR / f"openmeteo_daily_{date.today().isoformat()}.log"
    line = f"{utc_now().isoformat()} | {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_df_overwrite(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)

    if path.exists():
        path.unlink()

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        csv_path.unlink()

    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def read_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"No existe {path} ni {csv_path}")

def load_openmeteo_checkpoint() -> Dict[str, Any]:
    if not OPENMETEO_CHECKPOINT_PATH.exists():
        return {}
    with open(OPENMETEO_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_openmeteo_checkpoint(payload: Dict[str, Any]) -> None:
    ensure_dir(OPENMETEO_CHECKPOINT_PATH.parent)
    with open(OPENMETEO_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def clear_openmeteo_checkpoint() -> None:
    if OPENMETEO_CHECKPOINT_PATH.exists():
        OPENMETEO_CHECKPOINT_PATH.unlink()

def clear_openmeteo_inprogress() -> None:
    if OPENMETEO_INPROGRESS_SILVER_PATH.exists():
        OPENMETEO_INPROGRESS_SILVER_PATH.unlink()

    csv_path = OPENMETEO_INPROGRESS_SILVER_PATH.with_suffix(".csv")
    if csv_path.exists():
        csv_path.unlink()

def append_openmeteo_inprogress(df_new: pd.DataFrame) -> None:
    upsert_silver_append_dedupe(
        df_new=df_new,
        path=OPENMETEO_INPROGRESS_SILVER_PATH,
        key_cols=["estacion_codigo", "fecha"],
    )

def upsert_silver_append_dedupe(df_new: pd.DataFrame, path: Path, key_cols: List[str]) -> None:
    ensure_dir(path.parent)

    df_new = df_new.dropna(axis=1, how="all")
    if df_new.empty:
        return

    if path.exists():
        df_old = pd.read_parquet(path)
        common_cols = df_old.columns.union(df_new.columns)
        df_old = df_old.reindex(columns=common_cols)
        df_new = df_new.reindex(columns=common_cols)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new.copy()

    keys = [c for c in key_cols if c in df_all.columns]
    if keys:
        df_all = df_all.drop_duplicates(subset=keys, keep="last")

    df_all.to_parquet(path, index=False)


def _parse_station_date_series(series: pd.Series) -> pd.Series:
    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ms", errors="coerce", utc=True).dt.date

    s = s.astype("string").str.strip()
    numeric_mask = s.str.fullmatch(r"\d+")

    out = pd.to_datetime(s, errors="coerce", utc=True)

    if numeric_mask.any():
        out.loc[numeric_mask] = pd.to_datetime(
            s.loc[numeric_mask].astype("int64"),
            unit="ms",
            errors="coerce",
            utc=True,
        )

    return out.dt.date


def extract_invalid_daily_var(reason: str) -> Optional[str]:
    """
    Intenta extraer de un error 400 de Open-Meteo el nombre de la variable daily no soportada.
    Ejemplo típico:
    Cannot initialize WeatherVariable from invalid String value visibility_mean for key daily
    """
    if not reason:
        return None

    patterns = [
        r"invalid String value\s+([A-Za-z0-9_]+)\s+for key daily",
        r"invalid String value\s+([A-Za-z0-9_]+)\s+for key",
    ]

    for pattern in patterns:
        m = re.search(pattern, reason)
        if m:
            return m.group(1)

    return None


# -------------------------
# CARGA DE ESTACIONES DESDE GOLD SIAR
# -------------------------
def load_dim_estacion(include_closed: bool = False) -> pd.DataFrame:
    if not DIM_ESTACION_PATH.exists():
        raise FileNotFoundError(f"No existe {DIM_ESTACION_PATH}. Ejecuta antes el pipeline INFO de SIAR.")

    df = pd.read_parquet(DIM_ESTACION_PATH).copy()

    required_cols = {"estacion_codigo", "latitud", "longitud"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"dim_estacion no contiene las columnas necesarias: {sorted(missing)}")

    df["estacion_codigo"] = df["estacion_codigo"].astype("string").str.strip().str.upper()
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    if "altitud_m" in df.columns:
        df["altitud_m"] = pd.to_numeric(df["altitud_m"], errors="coerce")

    if "fecha_baja" in df.columns:
        df["fecha_baja"] = _parse_station_date_series(df["fecha_baja"])

    df = df.dropna(subset=["estacion_codigo", "latitud", "longitud"]).copy()

    if not include_closed and "fecha_baja" in df.columns:
        today = date.today()
        df = df[(df["fecha_baja"].isna()) | (df["fecha_baja"] >= today)].copy()

    df = df.sort_values(["estacion_codigo"]).drop_duplicates(subset=["estacion_codigo"], keep="first")

    keep_cols = [
        "estacion_id",
        "estacion_codigo",
        "estacion_nombre",
        "termino_municipal",
        "Provincia",
        "Codigo_Provincia",
        "CCAA",
        "Codigo_CCAA",
        "red_estacion",
        "altitud_m",
        "latitud",
        "longitud",
        "fecha_instalacion",
        "fecha_baja",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].reset_index(drop=True)


# -------------------------
# OPEN-METEO BRONZE
# -------------------------

def is_openmeteo_rate_limit_error(payload: Dict[str, Any], status_code: int) -> bool:
    if status_code != 429:
        return False
    reason = str(payload.get("reason", "")).lower()
    return "limit exceeded" in reason or "try again in one minute" in reason

def get_openmeteo_rate_limit_wait_seconds(error_text: str) -> int:
    txt = str(error_text).lower()

    if "next hour" in txt or "hourly api request limit exceeded" in txt:
        return 3605

    if "one minute" in txt or "minutely api request limit exceeded" in txt:
        return 65

    return 65

def fetch_openmeteo_daily_forecast(
    latitud: float,
    longitud: float,
    altitud_m: Optional[float] = None,
    forecast_days: int = 16,
    past_days: int = 0,
    timezone_str: str = "Europe/Madrid",
    timeout: int = DEFAULT_TIMEOUT,
    cell_selection: str = "land",
    daily_vars: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Lanza la consulta contra Open-Meteo.
    Si alguna variable daily no es soportada por el endpoint/modelo, la elimina y reintenta.
    """
    vars_to_request = list(daily_vars or OPENMETEO_DAILY_VARS_REQUESTED)

    while True:
        params: Dict[str, Any] = {
            "latitude": float(latitud),
            "longitude": float(longitud),
            "daily": ",".join(vars_to_request),
            "timezone": timezone_str,
            "forecast_days": int(forecast_days),
            "past_days": int(past_days),
            "cell_selection": cell_selection,
        }

        if altitud_m is not None and pd.notna(altitud_m):
            params["elevation"] = float(altitud_m)

        r = requests.get(OPENMETEO_BASE_URL, params=params, timeout=timeout)

        if r.ok:
            payload = r.json()
            if payload.get("error"):
                raise RuntimeError(f"Open-Meteo devolvió error lógico: {payload}")
            return payload, vars_to_request

        try:
            error_payload = r.json()
        except Exception:
            r.raise_for_status()
            raise RuntimeError("Error no JSON de Open-Meteo")

        reason = error_payload.get("reason", "")
        invalid_var = extract_invalid_daily_var(reason)

        if r.status_code == 400 and invalid_var and invalid_var in vars_to_request:
            vars_to_request.remove(invalid_var)
            log_line(f"[WARN] Open-Meteo no acepta daily='{invalid_var}'. Se reintenta sin esa variable.")
            if not vars_to_request:
                raise RuntimeError("No queda ninguna variable daily válida para solicitar a Open-Meteo.")
            continue

        if is_openmeteo_rate_limit_error(error_payload, r.status_code):
            reason_txt = str(error_payload.get("reason", "")).lower()

            if "next hour" in reason_txt or "hourly" in reason_txt:
                raise RuntimeError(f"[OPENMETEO_RATE_LIMIT_HOURLY] {error_payload}")

            if "one minute" in reason_txt or "minutely" in reason_txt:
                raise RuntimeError(f"[OPENMETEO_RATE_LIMIT_MINUTELY] {error_payload}")

            # fallback conservador
            raise RuntimeError(f"[OPENMETEO_RATE_LIMIT_UNKNOWN] {error_payload}")

        raise RuntimeError(f"Open-Meteo error HTTP {r.status_code}: {error_payload}")


def save_bronze_payload(
    payload: Dict[str, Any],
    estacion_codigo: str,
    request_utc: datetime,
    forecast_days: int,
    past_days: int,
) -> Path:
    ensure_dir(BRONZE_DIR)

    ts = request_utc.strftime("%Y%m%d_%H%M%S")
    path = BRONZE_DIR / f"{ts}_openmeteo_diarios_estacion_{estacion_codigo}_f{forecast_days}_p{past_days}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


# -------------------------
# OPEN-METEO SILVER
# -------------------------
def normalize_openmeteo_daily_to_silver(
    payload: Dict[str, Any],
    station_row: pd.Series,
    request_utc: datetime,
    accepted_daily_vars: List[str],
) -> pd.DataFrame:
    daily = payload.get("daily", {})
    daily_units = payload.get("daily_units", {})

    if not isinstance(daily, dict) or "time" not in daily:
        return pd.DataFrame()

    df = pd.DataFrame(daily).copy()
    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"time": "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date

    # Claves y metadatos de estación
    df.insert(0, "estacion_codigo", station_row["estacion_codigo"])

    for c in [
        "estacion_id",
        "estacion_nombre",
        "termino_municipal",
        "Provincia",
        "Codigo_Provincia",
        "CCAA",
        "Codigo_CCAA",
        "red_estacion",
        "altitud_m",
        "latitud",
        "longitud",
    ]:
        if c in station_row.index:
            df[c] = station_row.get(c)

    # Metadatos de forecast
    request_date = request_utc.date()
    df["request_generated_utc"] = request_utc.isoformat()
    df["forecast_reference_date"] = request_date
    df["horizonte_dias"] = df["fecha"].apply(
        lambda x: (x - request_date).days if pd.notna(x) else pd.NA
    )

    # Metadatos de respuesta Open-Meteo
    df["response_latitude"] = payload.get("latitude")
    df["response_longitude"] = payload.get("longitude")
    df["response_elevation"] = payload.get("elevation")
    df["utc_offset_seconds"] = payload.get("utc_offset_seconds")
    df["timezone"] = payload.get("timezone")
    df["timezone_abbreviation"] = payload.get("timezone_abbreviation")
    df["generationtime_ms"] = payload.get("generationtime_ms")

    df["accepted_daily_vars_json"] = json.dumps(accepted_daily_vars, ensure_ascii=False)
    df["daily_units_json"] = json.dumps(daily_units, ensure_ascii=False)

    # Tipado suave
    non_numeric_cols = {
        "fecha",
        "sunrise",
        "sunset",
        "request_generated_utc",
        "forecast_reference_date",
        "accepted_daily_vars_json",
        "daily_units_json",
        "timezone",
        "timezone_abbreviation",
        "estacion_codigo",
        "estacion_nombre",
        "termino_municipal",
        "Provincia",
        "Codigo_Provincia",
        "CCAA",
        "Codigo_CCAA",
        "red_estacion",
    }

    for c in df.columns:
        if c in non_numeric_cols:
            continue
        if c in {"response_latitude", "response_longitude", "response_elevation", "altitud_m", "latitud", "longitud"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            continue
        if c == "estacion_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")
            continue

        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

    return df


def run_openmeteo_bronze_silver(
    forecast_days: int = 16,
    past_days: int = 0,
    timezone_str: str = "Europe/Madrid",
    include_closed: bool = False,
    sleep_s: float = 0.10,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
) -> bool:
    stations = load_dim_estacion(include_closed=include_closed)
    log_line(f"[INFO] Estaciones válidas para Open-Meteo: {len(stations)}")

    run_signature = {
        "run_date": date.today().isoformat(),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone_str": timezone_str,
        "include_closed": include_closed,
    }

    checkpoint = load_openmeteo_checkpoint()

    same_run = (
        checkpoint.get("run_date") == run_signature["run_date"]
        and checkpoint.get("forecast_days") == run_signature["forecast_days"]
        and checkpoint.get("past_days") == run_signature["past_days"]
        and checkpoint.get("timezone_str") == run_signature["timezone_str"]
        and checkpoint.get("include_closed") == run_signature["include_closed"]
        and checkpoint.get("status") == "in_progress"
    )

    if same_run and OPENMETEO_INPROGRESS_SILVER_PATH.exists():
        start_idx = int(checkpoint.get("next_station_idx", 0))
        log_line(f"[INFO] Reanudando Open-Meteo desde estación índice {start_idx}.")
    else:
        start_idx = 0
        clear_openmeteo_checkpoint()
        clear_openmeteo_inprogress()
        save_openmeteo_checkpoint({
            **run_signature,
            "status": "in_progress",
            "next_station_idx": 0,
            "total_stations": len(stations),
            "started_utc": utc_now().isoformat(),
        })
        log_line("[INFO] Iniciando nueva ejecución Open-Meteo desde cero.")

    for original_idx, row in stations.iloc[start_idx:].iterrows():
        est = row["estacion_codigo"]
        request_utc = utc_now()

        attempt = 1
        while True:
            try:
                payload, accepted_vars = fetch_openmeteo_daily_forecast(
                    latitud=row["latitud"],
                    longitud=row["longitud"],
                    altitud_m=row.get("altitud_m"),
                    forecast_days=forecast_days,
                    past_days=past_days,
                    timezone_str=timezone_str,
                    timeout=timeout,
                    daily_vars=OPENMETEO_DAILY_VARS_REQUESTED,
                )

                bronze_path = save_bronze_payload(
                    payload=payload,
                    estacion_codigo=est,
                    request_utc=request_utc,
                    forecast_days=forecast_days,
                    past_days=past_days,
                )

                df_new = normalize_openmeteo_daily_to_silver(
                    payload=payload,
                    station_row=row,
                    request_utc=request_utc,
                    accepted_daily_vars=accepted_vars,
                )

                if not df_new.empty:
                    append_openmeteo_inprogress(df_new)

                log_line(
                    f"[OK] Open-Meteo est={est} filas={len(df_new)} bronze={bronze_path.name}"
                )

                save_openmeteo_checkpoint({
                    **run_signature,
                    "status": "in_progress",
                    "next_station_idx": int(original_idx) + 1,
                    "total_stations": len(stations),
                    "last_station_ok": est,
                    "last_success_utc": utc_now().isoformat(),
                })
                break

            except Exception as e:
                err_txt = str(e).lower()

                if "[openmeteo_rate_limit_hourly]" in err_txt:
                    wait_s = 3605
                    log_line(
                        f"[WAIT] Open-Meteo est={est} límite horario detectado. "
                        f"Esperando {wait_s}s y reintentando la misma estación."
                    )
                    time.sleep(wait_s)
                    attempt = 1
                    continue

                if "[openmeteo_rate_limit_minutely]" in err_txt:
                    wait_s = 65
                    log_line(
                        f"[WAIT] Open-Meteo est={est} límite por minuto detectado. "
                        f"Esperando {wait_s}s y reintentando la misma estación."
                    )
                    time.sleep(wait_s)
                    attempt = 1
                    continue

                if "[openmeteo_rate_limit_unknown]" in err_txt:
                    wait_s = 3605
                    log_line(
                        f"[WAIT] Open-Meteo est={est} límite no clasificado. "
                        f"Por seguridad se espera {wait_s}s y se reintenta."
                    )
                    time.sleep(wait_s)
                    attempt = 1
                    continue

                # Para otros errores, mantener reintentos normales
                if attempt >= max_retries:
                    log_line(f"[ERROR] Open-Meteo est={est} err={repr(e)}")
                    raise

                wait_s = 1.5 * attempt
                log_line(
                    f"[WARN] Open-Meteo est={est} intento={attempt}/{max_retries} "
                    f"err={repr(e)}. Esperando {wait_s}s."
                )
                time.sleep(wait_s)
                attempt += 1

            time.sleep(sleep_s)

    # Si ha llegado aquí, la corrida ha terminado completa.
    if OPENMETEO_INPROGRESS_SILVER_PATH.exists():
        df_silver_snapshot = read_table(OPENMETEO_INPROGRESS_SILVER_PATH).copy()

        df_silver_snapshot["estacion_codigo"] = (
            df_silver_snapshot["estacion_codigo"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
        df_silver_snapshot["fecha"] = pd.to_datetime(
            df_silver_snapshot["fecha"], errors="coerce"
        ).dt.date

        df_silver_snapshot = (
            df_silver_snapshot
            .sort_values(["estacion_codigo", "fecha", "request_generated_utc"])
            .drop_duplicates(subset=["estacion_codigo", "fecha"], keep="last")
            .reset_index(drop=True)
        )

        save_df_overwrite(df_silver_snapshot, SILVER_PATH)
        log_line(f"[OK] Silver Open-Meteo sobrescrito -> {SILVER_PATH.name} filas={len(df_silver_snapshot)}")

        clear_openmeteo_inprogress()
        clear_openmeteo_checkpoint()
        return True

    log_line("[WARN] No existe silver temporal de Open-Meteo. No se actualiza snapshot oficial.")
    return False

# -------------------------
# GOLD UNIFICADO
# -------------------------
def build_openmeteo_gold_rows(df_om_silver: pd.DataFrame, siar_gold_columns: List[str]) -> pd.DataFrame:
    df = df_om_silver.copy()

    # Normalización de claves
    df["estacion_codigo"] = df["estacion_codigo"].astype("string").str.strip().str.upper()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date

    # Fuente y tipo
    df["fuente_dato"] = "OPENMETEO"
    df["tipo_dato"] = "PREDICTIVO"
    df["forecast_run_utc"] = df.get("request_generated_utc", pd.NA)
    df["forecast_reference_date"] = pd.to_datetime(
        df.get("forecast_reference_date", pd.NA), errors="coerce"
    ).dt.date
    df["horizonte_dias"] = pd.to_numeric(df.get("horizonte_dias", pd.NA), errors="coerce")

    # Metadatos Open-Meteo que conservamos como om_*
    meta_rename = {
        "daily_units_json": "om_daily_units_json",
        "accepted_daily_vars_json": "om_requested_daily_vars_json",
        "response_latitude": "om_response_latitude",
        "response_longitude": "om_response_longitude",
        "response_elevation": "om_response_elevation",
        "utc_offset_seconds": "om_utc_offset_seconds",
        "timezone": "om_timezone",
        "timezone_abbreviation": "om_timezone_abbreviation",
        "generationtime_ms": "om_generationtime_ms",
    }
    for src, dst in meta_rename.items():
        if src in df.columns:
            df[dst] = df[src]

    # Mapeo a columnas canónicas SIAR
    for siar_col, om_col in OPENMETEO_TO_SIAR_CANONICAL_MAP.items():
        if om_col in df.columns:
            df[siar_col] = df[om_col]
    
    protected_cols = {
        "estacion_codigo",
        "fecha",
        "fuente_dato",
        "tipo_dato",
        "forecast_run_utc",
        "forecast_reference_date",
        "horizonte_dias",
    }

    # Conservamos también todas las columnas Open-Meteo con prefijo om_ en bloque
    om_source_cols = [
        c for c in df.columns
        if c not in protected_cols
        and not c.startswith("om_")
        and f"om_{c}" not in df.columns
    ]

    if om_source_cols:
        om_prefixed_df = df[om_source_cols].copy()
        om_prefixed_df.columns = [f"om_{c}" for c in om_source_cols]
        df = pd.concat([df, om_prefixed_df], axis=1)

    # Aseguramos que existan todas las columnas del gold SIAR actual en bloque
    missing_siar_cols = [col for col in siar_gold_columns if col not in df.columns]
    if missing_siar_cols:
        missing_df = pd.DataFrame({col: pd.NA for col in missing_siar_cols}, index=df.index)
        df = pd.concat([df, missing_df], axis=1)

    # Si en el fact SIAR ya existe gold_generated_utc, se recalculará al final
    extra_cols = UNIFIED_META_COLS + sorted([c for c in df.columns if c.startswith("om_")])

    keep_cols = list(siar_gold_columns)
    for c in extra_cols:
        if c not in keep_cols:
            keep_cols.append(c)

    keep_cols = list(dict.fromkeys(keep_cols))

    # defensa extra por si alguna columna vuelve a repetirse
    df = df.loc[:, ~df.columns.duplicated()].copy()

    df = df.reindex(columns=keep_cols)
    return df


def build_siar_gold_rows(df_siar_gold: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    df = df_siar_gold.copy()

    df["estacion_codigo"] = df["estacion_codigo"].astype("string").str.strip().str.upper()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date

    df["fuente_dato"] = "SIAR"
    df["tipo_dato"] = "OBSERVADO"
    df["forecast_run_utc"] = pd.NA
    df["forecast_reference_date"] = pd.NA
    df["horizonte_dias"] = pd.NA

    missing_cols = [c for c in target_columns if c not in df.columns]
    if missing_cols:
        missing_df = pd.DataFrame({c: pd.NA for c in missing_cols}, index=df.index)
        df = pd.concat([df, missing_df], axis=1)

    return df[target_columns].copy()

def merge_siar_with_openmeteo_context(
    df_siar_gold: pd.DataFrame,
    df_om_gold: pd.DataFrame,
    previous_context: Optional[pd.DataFrame],
    target_columns: List[str],
) -> pd.DataFrame:
    key_cols = ["estacion_codigo", "fecha"]

    df_siar_base = build_siar_gold_rows(df_siar_gold, target_columns)
    df_om_now = df_om_gold.reindex(columns=target_columns).copy()

    context_cols = [
        c for c in target_columns
        if c.startswith("om_") or c in {
            "forecast_run_utc",
            "forecast_reference_date",
            "horizonte_dias",
            "om_daily_units_json",
            "om_requested_daily_vars_json",
            "om_response_latitude",
            "om_response_longitude",
            "om_response_elevation",
            "om_utc_offset_seconds",
            "om_timezone",
            "om_timezone_abbreviation",
            "om_generationtime_ms",
        }
    ]

    if previous_context is not None and not previous_context.empty:
        prev = previous_context.copy()
        prev["estacion_codigo"] = prev["estacion_codigo"].astype("string").str.strip().str.upper()
        prev["fecha"] = pd.to_datetime(prev["fecha"], errors="coerce").dt.date
        prev = prev[[c for c in key_cols + context_cols if c in prev.columns]].copy()

        df_siar_base = df_siar_base.merge(
            prev,
            on=key_cols,
            how="left",
            suffixes=("", "_prevctx"),
        )

        for c in context_cols:
            prev_c = f"{c}_prevctx"
            if prev_c in df_siar_base.columns:
                df_siar_base[c] = df_siar_base[c].combine_first(df_siar_base[prev_c])
                df_siar_base = df_siar_base.drop(columns=[prev_c])

    merged = df_siar_base.merge(
        df_om_now,
        on=key_cols,
        how="outer",
        suffixes=("_siar", "_om"),
    )

    has_siar = merged.get("tipo_dato_siar", pd.Series(index=merged.index, dtype="object")).notna()
    has_om = merged.get("tipo_dato_om", pd.Series(index=merged.index, dtype="object")).notna()

    final_data = {
        "estacion_codigo": merged["estacion_codigo"],
        "fecha": merged["fecha"],
    }

    for c in target_columns:
        if c in {"estacion_codigo", "fecha", "fuente_dato", "tipo_dato"}:
            continue

        c_siar = f"{c}_siar"
        c_om = f"{c}_om"

        s_siar = merged[c_siar] if c_siar in merged.columns else pd.Series(pd.NA, index=merged.index)
        s_om = merged[c_om] if c_om in merged.columns else pd.Series(pd.NA, index=merged.index)

        # columnas canónicas/base -> preferencia SIAR
        # columnas om_* y metadatos forecast -> preferencia OPENMETEO actual, luego contexto previo
        if c.startswith("om_") or c in {
            "forecast_run_utc",
            "forecast_reference_date",
            "horizonte_dias",
            "om_daily_units_json",
            "om_requested_daily_vars_json",
            "om_response_latitude",
            "om_response_longitude",
            "om_response_elevation",
            "om_utc_offset_seconds",
            "om_timezone",
            "om_timezone_abbreviation",
            "om_generationtime_ms",
        }:
            final_data[c] = s_om.combine_first(s_siar)
        else:
            final_data[c] = s_siar.combine_first(s_om)

    final = pd.DataFrame(final_data)

    om_cols = [c for c in final.columns if c.startswith("om_")]
    has_om_context = final[om_cols].notna().any(axis=1) if om_cols else pd.Series(False, index=final.index)

    final["fuente_dato"] = "OPENMETEO"
    final.loc[has_siar & ~has_om_context, "fuente_dato"] = "SIAR"
    final.loc[has_siar & has_om_context, "fuente_dato"] = "SIAR+OPENMETEO"

    final["tipo_dato"] = "PREDICTIVO"
    final.loc[has_siar, "tipo_dato"] = "OBSERVADO"

    final["gold_generated_utc"] = utc_now().isoformat()

    missing_cols = [c for c in target_columns if c not in final.columns]
    if missing_cols:
        missing_df = pd.DataFrame({c: pd.NA for c in missing_cols}, index=final.index)
        final = pd.concat([final, missing_df], axis=1)

    return final[target_columns].copy()


def run_openmeteo_gold(
    datos_calculados_siar: bool = True,
    overwrite_siar_gold_fact: bool = True,
) -> Path:
    siar_gold_path = DATA_GOLD_DIR / f"fact_siar_diarios_estacion_calc{int(datos_calculados_siar)}.parquet"

    if not siar_gold_path.exists():
        raise FileNotFoundError(
            f"No existe {siar_gold_path}. Ejecuta antes run_datos_pipeline de SIAR."
        )

    if not SILVER_PATH.exists():
        raise FileNotFoundError(
            f"No existe {SILVER_PATH}. Ejecuta antes la parte Bronze/Silver de Open-Meteo."
        )

    df_siar = read_table(siar_gold_path).copy()
    if df_siar.empty:
        raise ValueError(f"El Gold SIAR está vacío: {siar_gold_path}")

    df_om_silver = read_table(SILVER_PATH).copy()
    if df_om_silver.empty:
        raise ValueError(f"El Silver Open-Meteo está vacío: {SILVER_PATH}")

    previous_context = None
    if MERGED_CONTEXT_PATH.exists():
        previous_context = read_table(MERGED_CONTEXT_PATH).copy()

    siar_gold_columns = df_siar.columns.tolist()
    df_om_gold = build_openmeteo_gold_rows(df_om_silver, siar_gold_columns)

    target_columns = siar_gold_columns[:]
    for c in UNIFIED_META_COLS:
        if c not in target_columns:
            target_columns.append(c)
    for c in sorted([c for c in df_om_gold.columns if c.startswith("om_")]):
        if c not in target_columns:
            target_columns.append(c)

    target_columns = list(dict.fromkeys(target_columns))
    df_om_gold = df_om_gold.loc[:, ~df_om_gold.columns.duplicated()].copy()

    df_all = merge_siar_with_openmeteo_context(
        df_siar_gold=df_siar,
        df_om_gold=df_om_gold,
        previous_context=previous_context,
        target_columns=target_columns,
    )

    output_path = (
        siar_gold_path
        if overwrite_siar_gold_fact
        else DATA_GOLD_DIR / "fact_siar_diarios_estacion_calc1_unificado.parquet"
    )

    save_df_overwrite(df_all, output_path)
    save_df_overwrite(df_all, MERGED_CONTEXT_PATH)

    log_line(f"[OK] GOLD unificado generado -> {output_path.name} filas={len(df_all)}")
    log_line(f"[OK] Contexto Open-Meteo persistido -> {MERGED_CONTEXT_PATH.name} filas={len(df_all)}")
    return output_path


# -------------------------
# ORQUESTACIÓN
# -------------------------
def run_openmeteo_pipeline(
    forecast_days: int = 16,
    past_days: int = 0,
    timezone_str: str = "Europe/Madrid",
    include_closed: bool = False,
    sleep_s: float = 0.10,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
    datos_calculados_siar: bool = True,
    overwrite_siar_gold_fact: bool = True,
) -> Path:
    log_line("[INFO] Iniciando pipeline Open-Meteo DAILY...")

    completed = run_openmeteo_bronze_silver(
        forecast_days=forecast_days,
        past_days=past_days,
        timezone_str=timezone_str,
        include_closed=include_closed,
        sleep_s=sleep_s,
        timeout=timeout,
        max_retries=max_retries,
    )

    if not completed:
        log_line("[WARN] Open-Meteo no completó el snapshot. No se reconstruye Gold.")
        return SILVER_PATH

    output = run_openmeteo_gold(
        datos_calculados_siar=datos_calculados_siar,
        overwrite_siar_gold_fact=overwrite_siar_gold_fact,
    )

    log_line(f"[INFO] Pipeline Open-Meteo DAILY finalizado -> {output}")
    return output