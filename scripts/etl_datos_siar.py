from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from calendar import monthrange

from scripts.common.settings import (
    SIAR_BASE_URL,
    DATA_BRONZE_DATOS_DIR,
    DATA_SILVER_DIR,
    DATA_GOLD_DIR,
    CHECKPOINT_SIAR_DIARIOS_ESTACION,
    LOGS_DIR,
    DIM_ESTACION_PATH,
)

import pandas as pd
import requests

BASE = SIAR_BASE_URL
BRONZE_DIR = DATA_BRONZE_DATOS_DIR
SILVER_DIR = DATA_SILVER_DIR
GOLD_DIR = DATA_GOLD_DIR
CHECKPOINT_PATH = CHECKPOINT_SIAR_DIARIOS_ESTACION
LOG_DIR = LOGS_DIR
FIRST_VALID_CACHE_PATH = CHECKPOINT_PATH.with_name("siar_diarios_estacion_first_valid.json")
BUDGET_REFRESH_EVERY_CHUNKS = 12

DEFAULT_TIMEOUT = 60
MIN_API_DATE = date(1999, 1, 1)


def utc_now() -> datetime:
    """
    Devuelve la fecha y hora actual en UTC sin microsegundos.

    Returns:
        datetime: Timestamp actual en zona horaria UTC.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def ensure_dir(p: Path) -> None:
    """
    Crea un directorio y sus carpetas padre si no existen.

    Args:
        p (Path): Ruta del directorio a crear.
    """
    p.mkdir(parents=True, exist_ok=True)


def load_checkpoint() -> Dict[str, str]:
    """
    Carga el fichero de checkpoint del proceso incremental.

    El checkpoint guarda, para cada estación y configuración de descarga,
    la última fecha procesada correctamente. Si el fichero no existe,
    devuelve un diccionario vacío.

    Returns:
        Dict[str, str]: Estado del checkpoint en formato clave -> fecha ISO.
    """
    if not CHECKPOINT_PATH.exists():
        return {}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(state: Dict[str, str]) -> None:
    """
    Guarda en disco el estado actual del checkpoint incremental.

    Args:
        state (Dict[str, str]): Diccionario con la última fecha procesada
            por clave de estación/proceso.
    """
    ensure_dir(CHECKPOINT_PATH.parent)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)

def load_json_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_map(path: Path, payload: Dict[str, str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

def get_safe_budget(accesos: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not accesos:
        return {
            "acc_min_left": 999999,
            "acc_day_left": 999999,
            "reg_min_left": 999999,
            "reg_day_left": 999999,
        }
    return remaining_budget(accesos)

def probe_chunk_acceptance(
    token: str,
    estacion_id: str,
    fecha_ini: str,
    fecha_fin: str,
    datos_calculados: bool,
) -> str:
    """
    Devuelve:
    - 'accepted' si la API acepta el rango (haya o no filas)
    - 'invalid_min_date' si la fecha inicial no es válida para esa estación
    """
    try:
        fetch_diarios_estacion(
            token=token,
            estacion_id=estacion_id,
            fecha_ini=fecha_ini,
            fecha_fin=fecha_fin,
            datos_calculados=datos_calculados,
            max_retries=3,
            wait_s=65,
        )
        return "accepted"
    except Exception as e:
        if is_min_date_api_error(e):
            return "invalid_min_date"
        raise

def discover_first_valid_chunk_start(
    token: str,
    estacion_id: str,
    d_station_start: date,
    d_station_end: date,
    datos_calculados: bool,
) -> Optional[date]:
    """
    Busca por bisección el primer chunk mensual que la API acepta para la estación.
    No busca el primer día con datos, sino el primer mes cuya fecha inicial ya no
    dispara el error de fecha mínima no válida.
    """
    chunks = month_chunks(d_station_start, d_station_end, step_months=1)
    if not chunks:
        return None

    low, high = 0, len(chunks) - 1
    found_idx: Optional[int] = None

    while low <= high:
        mid = (low + high) // 2
        a, b = chunks[mid]

        status = probe_chunk_acceptance(
            token=token,
            estacion_id=estacion_id,
            fecha_ini=a.isoformat(),
            fecha_fin=b.isoformat(),
            datos_calculados=datos_calculados,
        )

        if status == "invalid_min_date":
            low = mid + 1
        else:
            found_idx = mid
            high = mid - 1

    if found_idx is None:
        return None

    return chunks[found_idx][0]

def log_line(msg: str) -> None:
    """
    Escribe una línea de log en consola y en un fichero diario.

    El log se guarda en el directorio configurado en LOG_DIR y el nombre
    del fichero incluye la fecha actual.

    Args:
        msg (str): Mensaje a registrar.
    """
    ensure_dir(LOG_DIR)
    log_path = LOG_DIR / f"siar_diarios_{date.today().isoformat()}.log"
    line = f"{utc_now().isoformat()} | {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_accesos(
    token: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 5,
    wait_s: int = 15,
) -> Dict[str, Any]:
    """
    Consulta el endpoint Info/ACCESOS de SIAR para obtener límites y consumo.

    Si SIAR devuelve 403 o un error temporal, reintenta varias veces antes
    de devolver un diccionario vacío.
    """
    url = f"{BASE}/API/V1/Info/ACCESOS"

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params={"token": token}, timeout=timeout)

            if r.status_code == 403:
                log_line(
                    f"[WARN] Info/ACCESOS 403 intento={attempt}/{max_retries}. "
                    f"Esperando {wait_s}s antes de reintentar."
                )
                time.sleep(wait_s)
                continue

            r.raise_for_status()

            payload = r.json()
            datos = payload.get("datos", [])
            return datos[0] if isinstance(datos, list) and datos else {}

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_line(
                    f"[WARN] Error consultando Info/ACCESOS intento={attempt}/{max_retries} "
                    f"err={repr(e)}. Esperando {wait_s}s."
                )
                time.sleep(wait_s)
                continue

            log_line(
                f"[WARN] No se pudo consultar Info/ACCESOS tras {max_retries} intentos. "
                f"err={repr(e)}"
            )
            return {}


def remaining_budget(accesos: Dict[str, Any]) -> Dict[str, int]:
    """
    Calcula presupuesto restante tanto por minuto como por día.
    """
    max_acc_min = int(accesos.get("MaxAccesosMinuto", 0) or 0)
    used_acc_min = int(accesos.get("NumAccesosMinutoActual", 0) or 0)

    max_acc_day = int(accesos.get("MaxAccesosDia", 0) or 0)
    used_acc_day = int(accesos.get("NumAccesosDiaActual", 0) or 0)

    max_reg_min = int(accesos.get("MaxRegistrosMinuto", 0) or 0)
    used_reg_min = int(accesos.get("RegistrosAcumuladosMinuto", 0) or 0)

    max_reg_day = int(accesos.get("MaxRegistrosDia", 0) or 0)
    used_reg_day = int(accesos.get("RegistrosAcumuladosDia", 0) or 0)

    return {
        "acc_min_left": max(0, max_acc_min - used_acc_min) if max_acc_min else 999999,
        "acc_day_left": max(0, max_acc_day - used_acc_day) if max_acc_day else 999999,
        "reg_min_left": max(0, max_reg_min - used_reg_min) if max_reg_min else 999999,
        "reg_day_left": max(0, max_reg_day - used_reg_day) if max_reg_day else 999999,
        "used_acc_min": used_acc_min,
        "used_acc_day": used_acc_day,
        "used_reg_min": used_reg_min,
        "used_reg_day": used_reg_day,
        "max_acc_min": max_acc_min,
        "max_acc_day": max_acc_day,
        "max_reg_min": max_reg_min,
        "max_reg_day": max_reg_day,
    }

def is_minute_quota_error_text(text: str) -> bool:
    """
    Detecta límites SIAR por minuto, tanto de peticiones como de datos.
    """
    msg = (text or "").lower()

    return (
        "en un minuto" in msg
        and (
            "número de peticiones permitidas" in msg
            or "máximo de peticiones permitidas" in msg
            or "número máximo de datos permitidos" in msg
            or "máximo de datos permitidos" in msg
            or "rebasaría el número máximo de datos permitidos" in msg
        )
    )

def sanitize_query_dates(fecha_ini: str, fecha_fin: str) -> tuple[str, str]:
    """
    Ajusta y valida un intervalo de fechas antes de consultar SIAR.

    Reglas aplicadas:
    - La fecha inicial no puede ser anterior a MIN_API_DATE.
    - La fecha final no puede ser futura.
    - Si tras el ajuste el intervalo queda inválido, lanza ValueError.

    Args:
        fecha_ini (str): Fecha inicial en formato YYYY-MM-DD.
        fecha_fin (str): Fecha final en formato YYYY-MM-DD.

    Returns:
        tuple[str, str]: Fechas saneadas en formato ISO.

    Raises:
        ValueError: Si la fecha inicial queda posterior a la fecha final.
    """
    d_ini = date.fromisoformat(fecha_ini)
    d_fin = date.fromisoformat(fecha_fin)

    if d_ini < MIN_API_DATE:
        d_ini = MIN_API_DATE

    today = date.today()
    if d_fin > today:
        d_fin = today

    if d_ini > d_fin:
        raise ValueError(
            f"Intervalo inválido tras saneado: fecha_ini={d_ini.isoformat()} fecha_fin={d_fin.isoformat()}"
        )

    return d_ini.isoformat(), d_fin.isoformat()

def is_min_date_api_error(exc: Exception) -> bool:
    """
    Indica si una excepción corresponde al error de SIAR por fecha mínima autorizada.
    """
    msg = str(exc).lower()
    return "403 por fecha mínima" in msg or (
        "fecha inicial" in msg and "inferior" in msg and "autorizada" in msg
    )

# -------------------------
# BRONCE
# -------------------------
def fetch_diarios_estacion(
    token: str,
    estacion_id: str,
    fecha_ini: str,
    fecha_fin: str,
    datos_calculados: bool,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 6,
    wait_s: int = 65,
) -> Dict[str, Any]:
    """
    Descarga datos diarios de una estación SIAR para un intervalo de fechas.
    Reintenta automáticamente ante límites por minuto de peticiones o datos.
    """
    fecha_ini, fecha_fin = sanitize_query_dates(fecha_ini, fecha_fin)

    url = f"{BASE}/API/V1/Datos/Diarios/ESTACION"
    params = {
        "token": token,
        "Id": estacion_id,
        "FechaInicial": fecha_ini,
        "FechaFinal": fecha_fin,
        "DatosCalculados": "true" if datos_calculados else "false",
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_line(
                    f"[WARN] Error HTTP est={estacion_id} rango={fecha_ini}..{fecha_fin} "
                    f"intento={attempt}/{max_retries} err={repr(e)}. Esperando {wait_s}s."
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(
                f"[SIAR] Error de red est={estacion_id} fecha_ini={fecha_ini} "
                f"fecha_fin={fecha_fin} err={repr(e)}"
            ) from e

        if r.status_code == 403 and is_minute_quota_error_text(r.text):
            if attempt < max_retries:
                log_line(
                    f"[RATE] Límite SIAR por minuto est={estacion_id} "
                    f"rango={fecha_ini}..{fecha_fin} intento={attempt}/{max_retries}. "
                    f"Esperando {wait_s}s."
                )
                time.sleep(wait_s)
                continue

            raise RuntimeError(
                f"[SIAR] Excedidos reintentos por cuota/minuto "
                f"est={estacion_id} fecha_ini={fecha_ini} fecha_fin={fecha_fin} "
                f"body={r.text[:500]}"
            )

        if r.status_code == 403 and "fecha inicial" in r.text.lower() and "inferior" in r.text.lower():
            raise RuntimeError(f"[SIAR] 403 por fecha mínima: {r.text[:500]}")

        if not r.ok:
            raise RuntimeError(
                f"[SIAR] Error {r.status_code} est={estacion_id} "
                f"fecha_ini={fecha_ini} fecha_fin={fecha_fin} "
                f"url={r.url} body={r.text[:500]}"
            )

        return r.json()

    raise RuntimeError(
        f"[SIAR] Excedidos reintentos est={estacion_id} fecha_ini={fecha_ini} fecha_fin={fecha_fin}"
    )


def payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Convierte el bloque 'datos' de un payload JSON de SIAR en un DataFrame.

    Si la clave 'datos' no existe o no contiene una lista, devuelve un
    DataFrame vacío.

    Args:
        payload (Dict[str, Any]): Respuesta JSON de la API SIAR.

    Returns:
        pd.DataFrame: DataFrame con los registros del payload.
    """
    datos = payload.get("datos", [])
    if not isinstance(datos, list):
        return pd.DataFrame()
    return pd.DataFrame(datos)


def save_bronze_chunk(
    payload: Dict[str, Any],
    estacion_id: str,
    fecha_ini: str,
    fecha_fin: str,
    datos_calculados: bool,
) -> Path:
    """
    Guarda en Bronze un bloque JSON crudo descargado de la API SIAR.

    El nombre del fichero incluye timestamp de ingesta, estación, rango de
    fechas consultado y si los datos son calculados o no.

    Args:
        payload (Dict[str, Any]): Respuesta JSON a persistir.
        estacion_id (str): Identificador de la estación.
        fecha_ini (str): Fecha inicial consultada.
        fecha_fin (str): Fecha final consultada.
        datos_calculados (bool): Indicador de datos calculados.

    Returns:
        Path: Ruta completa del fichero JSON generado en Bronze.
    """
    ensure_dir(BRONZE_DIR)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    calc_tag = "calc1" if datos_calculados else "calc0"
    path = BRONZE_DIR / f"{ts}_diarios_estacion_{estacion_id}_{fecha_ini}_{fecha_fin}_{calc_tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# -------------------------
# SILVER
# -------------------------
def upsert_silver_append_dedupe(df_new: pd.DataFrame, path: Path, key_cols: List[str]) -> None:
    """
    Actualiza un parquet Silver añadiendo nuevos registros y eliminando duplicados.

    El proceso:
    - elimina columnas completamente vacías del DataFrame entrante,
    - alinea columnas entre el parquet existente y los nuevos datos,
    - concatena ambos conjuntos,
    - elimina duplicados según las claves indicadas,
    - conserva el último registro en caso de colisión.

    Args:
        df_new (pd.DataFrame): Nuevos datos a incorporar en Silver.
        path (Path): Ruta del parquet Silver.
        key_cols (List[str]): Columnas clave usadas para deduplicar.
    """
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


def load_silver_minimal(silver_path: Path, cols: List[str]) -> Optional[pd.DataFrame]:
    """
    Carga desde Silver solo un subconjunto mínimo de columnas.

    Está pensada para comprobaciones rápidas, por ejemplo verificar si un mes
    ya está completamente cargado, evitando leer el parquet completo.

    Args:
        silver_path (Path): Ruta del parquet Silver.
        cols (List[str]): Columnas a cargar.

    Returns:
        Optional[pd.DataFrame]: DataFrame con las columnas solicitadas o None
        si el fichero no existe.
    """
    if not silver_path.exists():
        return None
    return pd.read_parquet(silver_path, columns=[c for c in cols if c])


def save_df_overwrite(df: pd.DataFrame, path: Path) -> None:
    """
    Guarda un DataFrame sobrescribiendo la versión previa.
    Intenta parquet y si falla guarda CSV.
    """
    ensure_dir(path.parent)

    if path.exists():
        path.unlink()

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        csv_path.unlink()

    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(csv_path, index=False, encoding="utf-8")


def read_table(path: Path) -> pd.DataFrame:
    """
    Lee una tabla desde parquet o desde CSV alternativo.
    """
    if path.exists():
        return pd.read_parquet(path)

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"No existe {path} ni {csv_path}")


def _parse_station_date_series(series: pd.Series) -> pd.Series:
    """
    Convierte una serie de fechas de estación a date, soportando:
    - datetime/timestamp
    - strings ISO
    - enteros en milisegundos epoch
    """
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


def build_station_start_dates_from_info() -> Dict[str, Optional[str]]:
    """
    Lee dim_estacion.parquet y construye un diccionario
    estacion_codigo -> fecha_instalacion mínima (YYYY-MM-DD).
    """
    path = DIM_ESTACION_PATH

    if not path.exists():
        log_line(f"[WARN] No existe {path}. Se usará start_date global.")
        return {}

    df = read_table(path).copy()

    required_cols = {"estacion_codigo", "fecha_instalacion"}
    if not required_cols.issubset(df.columns):
        log_line(
            "[WARN] dim_estacion no contiene estacion_codigo y fecha_instalacion. "
            "Se usará start_date global."
        )
        return {}

    df["estacion_codigo"] = (
        df["estacion_codigo"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["fecha_instalacion"] = _parse_station_date_series(df["fecha_instalacion"])

    agg = (
        df.groupby("estacion_codigo", dropna=False)["fecha_instalacion"]
        .min()
        .reset_index()
    )

    out: Dict[str, Optional[str]] = {}
    for _, row in agg.iterrows():
        est = row["estacion_codigo"]
        fi = row["fecha_instalacion"]
        out[str(est).strip().upper()] = fi.isoformat() if pd.notna(fi) else None

    log_line(f"[INFO] Fechas de instalación cargadas para {len(out)} estaciones desde dim_estacion.")
    return out


def normalize_station_date_map(
    station_dates: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Optional[str]]:
    """
    Normaliza un diccionario estación -> fecha a claves en mayúsculas.

    Args:
        station_dates (Optional[Dict[str, Optional[str]]]): Diccionario original.

    Returns:
        Dict[str, Optional[str]]: Diccionario normalizado.
    """
    if not station_dates:
        return {}

    out: Dict[str, Optional[str]] = {}
    for k, v in station_dates.items():
        if k is None:
            continue
        out[str(k).strip().upper()] = v
    return out


# -------------------------
# GOLD
# -------------------------
def run_datos_gold(datos_calculados: bool = True) -> Dict[str, Path]:
    """
    Ejecuta la capa Gold para los datos diarios SIAR.

    Lee el parquet Silver de diarios por estación y genera un snapshot Gold
    manteniendo una estructura estable para consumo analítico.
    """
    silver_path = SILVER_DIR / f"siar_diarios_estacion_calc{int(datos_calculados)}.parquet"
    df = read_table(silver_path).copy()

    if df.empty:
        raise ValueError(f"El fichero Silver está vacío: {silver_path}")

    rename_map = {
        "Estacion": "estacion_codigo",
        "Fecha": "fecha",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if "estacion_codigo" in df.columns:
        df["estacion_codigo"] = df["estacion_codigo"].astype("string").str.strip().str.upper()

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date

    df["gold_generated_utc"] = utc_now().isoformat()

    sort_cols = [c for c in ["estacion_codigo", "fecha"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    gold_path = GOLD_DIR / f"fact_siar_diarios_estacion_calc{int(datos_calculados)}.parquet"
    save_df_overwrite(df, gold_path)

    log_line(f"[OK] GOLD generado -> {gold_path.name} filas={len(df)}")

    return {"FACT_DIARIOS_ESTACION": gold_path}


def is_month_complete_in_silver(
    df_silver: Optional[pd.DataFrame],
    estacion: str,
    month_start: date,
    month_end: date,
) -> bool:
    """
    Comprueba si una estación ya tiene cargados en Silver todos los días
    de un rango mensual dado.

    Args:
        df_silver (Optional[pd.DataFrame]): DataFrame mínimo cargado desde Silver.
        estacion (str): Identificador de la estación.
        month_start (date): Primer día del rango a comprobar.
        month_end (date): Último día del rango a comprobar.

    Returns:
        bool: True si existen registros para todos los días del rango y False
        en caso contrario.
    """
    if df_silver is None or df_silver.empty:
        return False

    if "Estacion" not in df_silver.columns or "Fecha" not in df_silver.columns:
        return False

    expected = set(
        (month_start + timedelta(days=i)).isoformat()
        for i in range((month_end - month_start).days + 1)
    )

    fechas = df_silver.loc[df_silver["Estacion"].astype(str).str.upper() == estacion, "Fecha"].astype(str)
    existing = set(fechas[fechas.isin(expected)].tolist())

    return expected.issubset(existing)


def month_chunks(start: date, end: date, step_months: int = 1) -> List[Tuple[date, date]]:
    """
    Divide un intervalo de fechas en bloques de meses naturales.

    Cada bloque se devuelve como una tupla (fecha_inicio, fecha_fin) ajustada
    al rango real solicitado.

    Args:
        start (date): Fecha inicial del intervalo global.
        end (date): Fecha final del intervalo global.
        step_months (int, optional): Número de meses por bloque.

    Returns:
        List[Tuple[date, date]]: Lista de rangos [ini, fin].
    """
    if start > end:
        return []

    out: List[Tuple[date, date]] = []
    cur = start.replace(day=1)

    while cur <= end:
        year = cur.year
        month = cur.month + step_months - 1

        while month > 12:
            month -= 12
            year += 1

        last_day = monthrange(year, month)[1]
        block_end = date(year, month, last_day)

        a = max(start, cur)
        b = min(end, block_end)

        if a <= b:
            out.append((a, b))

        next_month = cur.month + step_months
        next_year = cur.year

        while next_month > 12:
            next_month -= 12
            next_year += 1

        cur = date(next_year, next_month, 1)

    return out


def run_incremental_diarios_por_estacion(
    token: str,
    estaciones: List[str],
    station_bajas: Optional[Dict[str, Optional[str]]] = None,
    station_start_dates: Optional[Dict[str, Optional[str]]] = None,
    start_date: str = "1999-01-01",
    end_date: Optional[str] = None,
    datos_calculados: bool = False,
    min_access_buffer: int = 5,
    min_records_buffer: int = 2000,
    sleep_s: float = 0.3,
    rebuild_history: bool = False,
) -> None:
    """
    Ejecuta la carga incremental de datos diarios SIAR por estación.

    Flujo general:
    - carga el checkpoint previo,
    - calcula el rango pendiente por estación,
    - divide el periodo en bloques mensuales,
    - evita llamar a SIAR si el mes ya está completo en Silver,
    - controla cuotas diarias y por minuto mediante Info/ACCESOS,
    - descarga cada bloque en Bronze,
    - transforma el payload y actualiza Silver,
    - guarda el checkpoint tras cada bloque correcto.

    Si rebuild_history=True:
    - se ignora el checkpoint previo,
    - el inicio de cada estación se calcula con su fecha_instalacion
      cuando esté disponible,
    - si no existe fecha_instalacion, se usa start_date global.

    Args:
        token (str): Token de autenticación para la API SIAR.
        estaciones (List[str]): Lista de identificadores de estación.
        station_bajas (Optional[Dict[str, Optional[str]]], optional):
            Diccionario estación -> fecha_baja en formato ISO.
        station_start_dates (Optional[Dict[str, Optional[str]]], optional):
            Diccionario estación -> fecha_instalacion en formato ISO.
        start_date (str, optional): Fecha inicial global en formato YYYY-MM-DD.
        end_date (Optional[str], optional): Fecha final global. Si no se indica,
            se usa ayer.
        datos_calculados (bool, optional): Indica si se descargan datos calculados.
        min_access_buffer (int, optional): Umbral mínimo de accesos restantes por día.
        min_records_buffer (int, optional): Umbral mínimo de registros restantes por día.
        sleep_s (float, optional): Pausa entre peticiones correctas.
        rebuild_history (bool, optional): Ignora checkpoint y rehace histórico.

    Returns:
        None
    """
    state = load_checkpoint()
    first_valid_cache = load_json_map(FIRST_VALID_CACHE_PATH)

    if end_date is None:
        end_date = str(date.today() - timedelta(days=1))

    d_global_start = date.fromisoformat(start_date)
    if d_global_start < MIN_API_DATE:
        d_global_start = MIN_API_DATE

    d_end = date.fromisoformat(end_date)

    station_bajas_norm = normalize_station_date_map(station_bajas)
    station_start_dates_norm = normalize_station_date_map(station_start_dates)

    silver_path = SILVER_DIR / f"siar_diarios_estacion_calc{int(datos_calculados)}.parquet"
    df_silver_cache = load_silver_minimal(silver_path, cols=["Fecha", "Estacion"])

    key_cols = ["Fecha", "Estacion"]

    for est in estaciones:
        est = str(est).strip().upper()
        ck_key = f"Diarios|ESTACION|{est}|calc={str(datos_calculados).lower()}"

        d_station_start = d_global_start
        fi = station_start_dates_norm.get(est)
        if fi:
            fi_date = date.fromisoformat(fi)
            if fi_date > d_station_start:
                d_station_start = fi_date

        d_station_end = d_end

        fb = station_bajas_norm.get(est)
        if fb:
            fb_date = date.fromisoformat(fb)

            if fb_date < d_station_start:
                state[ck_key] = d_end.isoformat()
                save_checkpoint(state)
                log_line(
                    f"[SKIP] est={est} motivo=fecha_baja<{d_station_start.isoformat()} "
                    f"fecha_baja={fb} -> no se extrae"
                )
                continue

            if fb_date < d_station_end:
                d_station_end = fb_date

        last = None if rebuild_history else state.get(ck_key)

        if last:
            d_start = date.fromisoformat(last) + timedelta(days=1)
        else:
            d_start = d_station_start

        cached_first_valid = first_valid_cache.get(est)
        if cached_first_valid:
            d_cached = date.fromisoformat(cached_first_valid)
            if d_cached > d_start:
                d_start = d_cached

        # IMPORTANTE: comprobar rango antes de descubrir primer chunk válido
        if d_start > d_station_end:
            log_line(
                f"[SKIP] est={est} motivo=sin_rango_pre_discovery "
                f"start={d_start.isoformat()} end={d_station_end.isoformat()} "
                f"fecha_instalacion={fi} fecha_baja={fb}"
            )
            continue

        if not cached_first_valid:
            discovered = discover_first_valid_chunk_start(
                token=token,
                estacion_id=est,
                d_station_start=d_start,
                d_station_end=d_station_end,
                datos_calculados=datos_calculados,
            )

            if discovered is None:
                state[ck_key] = d_station_end.isoformat()
                save_checkpoint(state)
                log_line(
                    f"[SKIP] est={est} motivo=sin_chunk_aceptado_por_api "
                    f"start={d_start.isoformat()} end={d_station_end.isoformat()}"
                )
                continue

            first_valid_cache[est] = discovered.isoformat()
            save_json_map(FIRST_VALID_CACHE_PATH, first_valid_cache)

            if discovered > d_start:
                d_start = discovered

        if d_start > d_station_end:
            log_line(
                f"[SKIP] est={est} motivo=sin_rango "
                f"start={d_start.isoformat()} end={d_station_end.isoformat()}"
            )
            continue

        log_line(
            f"[INFO] est={est} inicio_real={d_start.isoformat()} "
            f"fin_real={d_station_end.isoformat()} "
            f"fecha_instalacion={fi} fecha_baja={fb} rebuild_history={rebuild_history}"
        )

        chunks = month_chunks(d_start, d_station_end, step_months=1)

        budget = None
        chunks_since_budget_refresh = BUDGET_REFRESH_EVERY_CHUNKS

        for a, b in chunks:
            if is_month_complete_in_silver(df_silver_cache, est, a, b):
                state[ck_key] = b.isoformat()
                save_checkpoint(state)
                log_line(
                    f"[SKIP] est={est} rango={a.isoformat()}..{b.isoformat()} motivo=month_complete"
                )
                continue

            fecha_ini = a.isoformat()
            fecha_fin = b.isoformat()

            try:
                fecha_ini, fecha_fin = sanitize_query_dates(fecha_ini, fecha_fin)
            except ValueError as ve:
                log_line(
                    f"[SKIP] est={est} rango_original={a.isoformat()}..{b.isoformat()} motivo={repr(ve)}"
                )
                continue

            expected_rows = (
                date.fromisoformat(fecha_fin) - date.fromisoformat(fecha_ini)
            ).days + 1

            if budget is None or chunks_since_budget_refresh >= BUDGET_REFRESH_EVERY_CHUNKS:
                accesos = get_accesos(token)
                if not accesos:
                    log_line("[WARN] No se pudo verificar Info/ACCESOS. Se continúa con prudencia.")
                budget = get_safe_budget(accesos)
                chunks_since_budget_refresh = 0

            if (
                budget["acc_day_left"] <= min_access_buffer
                or budget["reg_day_left"] <= min_records_buffer
            ):
                log_line(
                    f"[STOP] quota_day_low "
                    f"acc_day_left={budget['acc_day_left']} "
                    f"reg_day_left={budget['reg_day_left']}"
                )
                save_checkpoint(state)
                return

            if (
                budget["acc_min_left"] <= 1
                or budget["reg_min_left"] < expected_rows
            ):
                wait_s = 65
                log_line(
                    f"[WAIT] cuota_minuto est={est} rango={fecha_ini}..{fecha_fin} "
                    f"acc_min_left={budget['acc_min_left']} "
                    f"reg_min_left={budget['reg_min_left']} "
                    f"expected_rows={expected_rows}. Esperando {wait_s}s."
                )
                time.sleep(wait_s)

                accesos = get_accesos(token)
                budget = get_safe_budget(accesos)

            acc_left = budget["acc_day_left"]
            reg_left = budget["reg_day_left"]

            try:
                payload = fetch_diarios_estacion(
                    token=token,
                    estacion_id=est,
                    fecha_ini=fecha_ini,
                    fecha_fin=fecha_fin,
                    datos_calculados=datos_calculados,
                )

                bronze_path = save_bronze_chunk(
                    payload=payload,
                    estacion_id=est,
                    fecha_ini=fecha_ini,
                    fecha_fin=fecha_fin,
                    datos_calculados=datos_calculados,
                )

                df = payload_to_df(payload)
                if not df.empty:
                    df["ingestion_utc"] = utc_now().isoformat()
                    upsert_silver_append_dedupe(df, silver_path, key_cols)

                df_silver_cache = load_silver_minimal(silver_path, cols=["Fecha", "Estacion"])

                state[ck_key] = fecha_fin
                save_checkpoint(state)

                log_line(
                    f"[OK] est={est} rango={fecha_ini}..{fecha_fin} filas={len(df)} "
                    f"bronze={bronze_path.name} accesos_left={acc_left} registros_left={reg_left}"
                )

                chunks_since_budget_refresh += 1

                # consumo aproximado local para no reconsultar accesos a cada mes
                budget["acc_day_left"] = max(0, budget["acc_day_left"] - 1)
                budget["acc_min_left"] = max(0, budget["acc_min_left"] - 1)
                budget["reg_day_left"] = max(0, budget["reg_day_left"] - len(df))
                budget["reg_min_left"] = max(0, budget["reg_min_left"] - len(df))

                time.sleep(sleep_s)

            except Exception as e:
                if is_min_date_api_error(e):
                    state[ck_key] = fecha_fin
                    save_checkpoint(state)

                    log_line(
                        f"[WARN] est={est} rango={fecha_ini}..{fecha_fin} "
                        f"motivo=fecha_minima_api_no_valida -> se salta chunk y se continúa"
                    )
                    continue

                save_checkpoint(state)
                log_line(f"[ERROR] est={est} rango={fecha_ini}..{fecha_fin} err={repr(e)}")
                raise

    save_checkpoint(state)
    log_line("[DONE] Incremental completo.")


def run_datos_pipeline(
    token: str,
    estaciones: List[str],
    station_bajas: Optional[Dict[str, Optional[str]]] = None,
    start_date: str = "1999-01-01",
    end_date: Optional[str] = None,
    datos_calculados: bool = True,
    min_access_buffer: int = 5,
    min_records_buffer: int = 2000,
    sleep_s: float = 0.3,
    rebuild_history: bool = False,
) -> None:
    """
    Orquesta la carga de datos diarios SIAR:
    Bronze -> Silver incremental -> Gold snapshot

    Si rebuild_history=True:
    - el inicio de cada estación se calcula con fecha_instalacion,
    - se ignora el checkpoint.
    """
    log_line("[INFO] Iniciando pipeline SIAR DATOS...")

    station_start_dates = build_station_start_dates_from_info()

    run_incremental_diarios_por_estacion(
        token=token,
        estaciones=estaciones,
        station_bajas=station_bajas,
        station_start_dates=station_start_dates,
        start_date=start_date,
        end_date=end_date,
        datos_calculados=datos_calculados,
        min_access_buffer=min_access_buffer,
        min_records_buffer=min_records_buffer,
        sleep_s=sleep_s,
        rebuild_history=rebuild_history,
    )

    g = run_datos_gold(datos_calculados=datos_calculados)
    for t, p in g.items():
        log_line(f"[OK] GOLD snapshot {t} -> {p}")

    log_line("[INFO] Pipeline SIAR DATOS finalizado.")