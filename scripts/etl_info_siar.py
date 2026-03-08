from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from scripts.common.settings import (
    SIAR_BASE_URL,
    DATA_BRONZE_INFO_DIR,
    DATA_SILVER_DIR,
    DATA_GOLD_DIR,
)

import json
import pandas as pd
import requests
import urllib.parse

BASE = SIAR_BASE_URL

BRONZE_DIR = DATA_BRONZE_INFO_DIR
SILVER_DIR = DATA_SILVER_DIR
GOLD_DIR = DATA_GOLD_DIR

# -------------------------
# Helpers comunes
# -------------------------
def utc_now() -> datetime:
    """
    Devuelve la fecha y hora actual en UTC sin microsegundos.

    Returns:
        datetime: Timestamp actual en UTC.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_now_iso() -> str:
    """
    Devuelve la fecha y hora actual en UTC en formato ISO 8601.

    Returns:
        str: Timestamp actual en formato ISO.
    """
    return utc_now().isoformat()


def make_run_id(ts: datetime | None = None) -> str:
    """
    Genera un identificador de ejecución estable en UTC para nombres de fichero.

    Si no se proporciona timestamp, utiliza el momento actual en UTC.

    Args:
        ts (datetime | None, optional): Timestamp de referencia.

    Returns:
        str: Identificador de ejecución con formato YYYYMMDD.
    """
    ts = ts or utc_now()
    return ts.strftime("%Y%m%d")


def save_json(payload: dict, path: Path) -> None:
    """
    Guarda un diccionario como fichero JSON en la ruta indicada.

    Crea automáticamente los directorios padre si no existen.

    Args:
        payload (dict): Contenido JSON a guardar.
        path (Path): Ruta de destino del fichero.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    """
    Lee un fichero JSON desde disco y devuelve su contenido.

    Args:
        path (Path): Ruta del fichero JSON.

    Returns:
        dict: Contenido del fichero parseado como diccionario.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_df_overwrite(df: pd.DataFrame, path: Path) -> None:
    """
    Guarda un DataFrame sobrescribiendo cualquier versión previa.

    El proceso elimina primero el parquet o CSV existente para evitar residuos
    de ejecuciones anteriores. Intenta guardar en parquet y, si falla,
    utiliza CSV como alternativa.

    Args:
        df (pd.DataFrame): DataFrame a persistir.
        path (Path): Ruta principal de salida en formato parquet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

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
    Lee una tabla desde parquet o, si no existe, desde CSV alternativo.

    Primero intenta leer el parquet indicado. Si no existe, busca un CSV con
    el mismo nombre base.

    Args:
        path (Path): Ruta esperada del parquet.

    Returns:
        pd.DataFrame: Tabla cargada en memoria.

    Raises:
        FileNotFoundError: Si no existe ni el parquet ni el CSV alternativo.
    """
    if path.exists():
        # parquet
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No existe {path} ni {csv_path}")


# -------------------------
# BRONZE (RAW histórico)
# -------------------------
def get_info_raw(token: str, tipo: str, timeout: int = 30) -> dict:
    """
    Descarga el JSON original del endpoint Info/{tipo} de la API SIAR.

    Args:
        token (str): Token de autenticación para la API.
        tipo (str): Tipo de recurso de información a consultar
            (por ejemplo, CCAA, PROVINCIAS o ESTACIONES).
        timeout (int, optional): Tiempo máximo de espera de la petición HTTP.

    Returns:
        dict: Respuesta JSON original del endpoint solicitado.

    Raises:
        RuntimeError: Si la API devuelve un error HTTP o una respuesta no JSON.
    """
    tipo = tipo.upper().strip()
    url = f"{BASE}/API/V1/Info/{tipo}"

    r = requests.get(url, params={"token": token}, timeout=timeout)

    if not r.ok:
        raise RuntimeError(f"[SIAR] Error {r.status_code} en Info/{tipo}: {r.text[:500]}")

    try:
        return r.json()
    except ValueError:
        # Respuesta no-JSON (a veces pasa en APIs con errores intermedios)
        raise RuntimeError(f"[SIAR] Respuesta no JSON en Info/{tipo}: {r.text[:500]}")


def extract_info_raw(token: str, tipos: List[str], timeout: int = 30) -> Dict[str, dict]:
    """
    Descarga varios endpoints de información SIAR y devuelve sus payloads raw.

    Args:
        token (str): Token de autenticación para la API.
        tipos (List[str]): Lista de tipos de información a consultar.
        timeout (int, optional): Tiempo máximo de espera por petición.

    Returns:
        Dict[str, dict]: Diccionario tipo -> payload JSON original.
    """
    out: Dict[str, dict] = {}
    for t in tipos:
        t_up = t.upper().strip()
        out[t_up] = get_info_raw(token=token, tipo=t_up, timeout=timeout)
    return out


def run_info_bronze(
    token: str,
    tipos: List[str] | None = None,
    timeout: int = 30,
    write_latest_copy: bool = True,
) -> Dict[str, Path]:
    """
    Ejecuta la capa Bronze del pipeline de información SIAR.

    Para cada tipo solicitado:
    - guarda una copia histórica inmutable con run_id,
    - opcionalmente actualiza una copia latest_* para facilitar la carga Silver.

    Args:
        token (str): Token de autenticación para la API.
        tipos (List[str] | None, optional): Lista de tipos a descargar.
            Si no se indica, usa CCAA, PROVINCIAS y ESTACIONES.
        timeout (int, optional): Tiempo máximo de espera por petición.
        write_latest_copy (bool, optional): Si es True, guarda también una
            copia mutable latest_*.

    Returns:
        Dict[str, Path]: Diccionario tipo -> ruta del fichero histórico generado.
    """
    if tipos is None:
        tipos = ["CCAA", "PROVINCIAS", "ESTACIONES"]

    run_ts = utc_now()
    run_id = make_run_id(run_ts)

    payloads = extract_info_raw(token=token, tipos=tipos, timeout=timeout)

    written: Dict[str, Path] = {}
    for tipo, payload in payloads.items():
        tipo_low = tipo.lower()

        # Histórico (inmutable)
        hist_path = BRONZE_DIR / f"{run_id}_siar_info_{tipo_low}.json"
        save_json(payload, hist_path)
        written[tipo] = hist_path

        # Copia "latest" (mutable) para facilitar la carga silver
        if write_latest_copy:
            latest_path = BRONZE_DIR / f"latest_siar_info_{tipo_low}.json"
            save_json(payload, latest_path)

    return written


def load_latest_bronze_info() -> Dict[str, dict]:
    """
    Carga los ficheros latest_* generados en Bronze para la capa de información.

    Returns:
        Dict[str, dict]: Diccionario con los payloads latest de CCAA,
        PROVINCIAS y ESTACIONES.

    Raises:
        FileNotFoundError: Si falta alguno de los ficheros latest requeridos.
    """
    files = {
        "CCAA": BRONZE_DIR / "latest_siar_info_ccaa.json",
        "PROVINCIAS": BRONZE_DIR / "latest_siar_info_provincias.json",
        "ESTACIONES": BRONZE_DIR / "latest_siar_info_estaciones.json",
    }

    out: Dict[str, dict] = {}
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(
                f"No existe {p}. Ejecuta primero run_info_bronze con write_latest_copy=True."
            )
        out[k] = read_json(p)

    return out


# -------------------------
# SILVER (snapshot overwrite)
# -------------------------
def payload_to_df(payload: dict) -> pd.DataFrame:
    """
    Convierte el bloque 'datos' de un payload JSON en un DataFrame.

    Args:
        payload (dict): Payload JSON con una clave 'datos' de tipo lista.

    Returns:
        pd.DataFrame: DataFrame construido a partir de payload['datos'].

    Raises:
        ValueError: Si la clave 'datos' no existe como lista.
    """
    datos = payload.get("datos", [])
    if not isinstance(datos, list):
        raise ValueError("El payload no tiene 'datos' como lista.")
    return pd.DataFrame(datos)


def run_info_silver() -> Dict[str, Path]:
    """
    Ejecuta la capa Silver del pipeline de información SIAR.

    Flujo:
    - carga los latest de Bronze,
    - transforma y normaliza CCAA, Provincias y Estaciones,
    - construye la tabla de territorio mediante join,
    - añade metadatos de ingesta,
    - guarda snapshots overwrite en `data/silver`.

    Returns:
        Dict[str, Path]: Diccionario con las rutas de salida de TERRITORIO y
        ESTACIONES en Silver.
    """
    payloads = load_latest_bronze_info()
    ingestion = utc_now_iso()

    # --- CCAA (raw -> normalizado mínimo) ---
    df_ccaa = payload_to_df(payloads["CCAA"]).copy()
    df_ccaa = df_ccaa.rename(columns={"CCAA": "CCAA", "Codigo": "Codigo_CCAA"})
    df_ccaa["CCAA"] = df_ccaa["CCAA"].astype("string").str.strip()
    df_ccaa["Codigo_CCAA"] = df_ccaa["Codigo_CCAA"].astype("string").str.strip().str.upper()
    df_ccaa = df_ccaa.drop_duplicates(subset=["Codigo_CCAA"], keep="first").reset_index(drop=True)

    # --- Provincias (raw -> normalizado mínimo) ---
    df_prov = payload_to_df(payloads["PROVINCIAS"]).copy()
    df_prov = df_prov.rename(
        columns={
            "Provincia": "Provincia",
            "Codigo": "Codigo_Provincia",
            "Codigo_CCAA": "Codigo_CCAA",
        }
    )
    df_prov["Provincia"] = df_prov["Provincia"].astype("string").str.strip()
    df_prov["Codigo_Provincia"] = df_prov["Codigo_Provincia"].astype("string").str.strip().str.upper()
    df_prov["Codigo_CCAA"] = df_prov["Codigo_CCAA"].astype("string").str.strip().str.upper()
    df_prov = df_prov.drop_duplicates(subset=["Codigo_Provincia"], keep="first").reset_index(drop=True)

    # --- TERRITORIO (join por Codigo_CCAA) ---
    # provincias.Codigo_CCAA  <->  ccaa.Codigo_CCAA
    df_territorio = df_prov.merge(
        df_ccaa[["Codigo_CCAA", "CCAA"]],
        on="Codigo_CCAA",
        how="left",
        validate="many_to_one",   # muchas provincias -> una CCAA
    )

    # Reordenar columnas como quieres
    df_territorio = df_territorio[["CCAA", "Codigo_CCAA", "Provincia", "Codigo_Provincia"]]

    # Chequeo rápido opcional: si hay provincias sin match de CCAA
    # (no revienta, pero te avisa)
    missing = df_territorio["CCAA"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} provincias sin match de CCAA en el join (Codigo_CCAA).")

    # Metadatos
    df_territorio["ingestion_utc"] = ingestion

    # Orden estable
    df_territorio = df_territorio.sort_values(["Codigo_CCAA", "Codigo_Provincia"]).reset_index(drop=True)

    # --- Estaciones (igual que antes) ---
    df_est = payload_to_df(payloads["ESTACIONES"]).copy()
    df_est = df_est.rename(
        columns={
            "Estacion": "estacion_nombre",
            "Codigo": "estacion_codigo",
            "Termino": "termino_municipal",
            "Longitud": "longitud_raw",
            "Latitud": "latitud_raw",
            "Altitud": "altitud_m",
            "XUTM": "utm_x",
            "YUTM": "utm_y",
            "Huso": "utm_huso",
            "Fecha_Instalacion": "fecha_instalacion",
            "Fecha_Baja": "fecha_baja",
            "Red_Estacion": "red_estacion",
        }
    )

    for c in ["estacion_nombre", "estacion_codigo", "termino_municipal", "longitud_raw", "latitud_raw", "red_estacion"]:
        if c in df_est.columns:
            df_est[c] = df_est[c].astype("string").str.strip()

    if "estacion_codigo" in df_est.columns:
        df_est["estacion_codigo"] = df_est["estacion_codigo"].astype("string").str.upper()

    for c in ["fecha_instalacion", "fecha_baja"]:
        if c in df_est.columns:
            df_est[c] = pd.to_datetime(df_est[c], errors="coerce", utc=True)

    for c in ["altitud_m", "utm_x", "utm_y", "utm_huso"]:
        if c in df_est.columns:
            df_est[c] = pd.to_numeric(df_est[c], errors="coerce")

    df_est["ingestion_utc"] = ingestion
    df_est = df_est.drop_duplicates(subset=["estacion_codigo"], keep="first").reset_index(drop=True)

    # --- Guardado (overwrite) ---
    paths = {
        "TERRITORIO": SILVER_DIR / "siar_territorio.parquet",
        "ESTACIONES": SILVER_DIR / "siar_estaciones.parquet",
    }

    save_df_overwrite(df_territorio, paths["TERRITORIO"])
    save_df_overwrite(df_est, paths["ESTACIONES"])

    return paths

# -------------------------
# GOLD (snapshot overwrite) -> mismo modelo que Silver
# -------------------------
def run_info_gold() -> Dict[str, Path]:
    """
    Ejecuta la capa Gold del pipeline de información SIAR.

    Lee las tablas Silver y construye:
    - una dimensión de territorio con identificador artificial,
    - una dimensión de estaciones con identificador artificial y columnas
      seleccionadas para consumo final.

    Returns:
        Dict[str, Path]: Diccionario con las rutas de salida de DIM_TERRITORIO
        y DIM_ESTACION en Gold.
    """
    df_terr = read_table(SILVER_DIR / "siar_territorio.parquet")
    df_est = read_table(SILVER_DIR / "siar_estaciones.parquet")

    # DIM_TERRITORIO: mismo contenido + id
    dim_terr = df_terr.copy()
    dim_terr = dim_terr.drop_duplicates(subset=["Codigo_Provincia"], keep="first").reset_index(drop=True)
    dim_terr = dim_terr.sort_values(["Codigo_CCAA", "Codigo_Provincia"]).reset_index(drop=True)
    dim_terr.insert(0, "territorio_id", range(1, len(dim_terr) + 1))

    # DIM_ESTACION: igual que antes
    dim_est = df_est.sort_values(["estacion_codigo"]).reset_index(drop=True)
    dim_est.insert(0, "estacion_id", range(1, len(dim_est) + 1))

    keep_est = [
        "estacion_id",
        "estacion_codigo",
        "estacion_nombre",
        "termino_municipal",
        "red_estacion",
        "altitud_m",
        "utm_huso",
        "utm_x",
        "utm_y",
        "fecha_instalacion",
        "fecha_baja",
    ]
    keep_est = [c for c in keep_est if c in dim_est.columns]
    dim_est = dim_est[keep_est]

    paths = {
        "DIM_TERRITORIO": GOLD_DIR / "dim_territorio.parquet",
        "DIM_ESTACION": GOLD_DIR / "dim_estacion.parquet",
    }

    save_df_overwrite(dim_terr, paths["DIM_TERRITORIO"])
    save_df_overwrite(dim_est, paths["DIM_ESTACION"])

    return paths

# -------------------------
# Orquestación
# -------------------------
def run_info_pipeline(token: str) -> None:
    """
    Orquesta la ejecución completa del pipeline SIAR de información.

    Flujo:
    1. Ejecuta Bronze y guarda histórico + latest.
    2. Ejecuta Silver a partir de los latest de Bronze.
    3. Ejecuta Gold a partir de las tablas Silver.
    4. Muestra por consola las rutas generadas en cada capa.

    Args:
        token (str): Token de autenticación para la API SIAR.

    Returns:
        None
    """
    print("[INFO] Iniciando pipeline SIAR INFO...")
    b = run_info_bronze(token=token, write_latest_copy=True)
    for t, p in b.items():
        print(f"[OK] BRONZE histórico {t} -> {p}")

    s = run_info_silver()
    for t, p in s.items():
        print(f"[OK] SILVER snapshot {t} -> {p}")

    g = run_info_gold()
    for t, p in g.items():
        print(f"[OK] GOLD snapshot {t} -> {p}")
    print("[INFO] Pipeline SIAR INFO finalizado.")