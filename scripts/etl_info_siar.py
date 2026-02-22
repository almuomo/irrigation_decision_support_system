from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import json
import pandas as pd
import requests
import urllib.parse

BASE = "https://servicio.mapa.gob.es/siarapi"

BRONZE_DIR = Path("data/bronce/info")
SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")


# -------------------------
# Helpers comunes
# -------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def make_run_id(ts: datetime | None = None) -> str:
    """
    ID de ejecución estable para nombres de ficheros (UTC).
    Ej: 20260222_153210
    """
    ts = ts or utc_now()
    return ts.strftime("%Y%m%d")


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_df_overwrite(df: pd.DataFrame, path: Path) -> None:
    """
    Overwrite defensivo: borra parquet/csv previo para evitar residuos.
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
    Devuelve el JSON original del endpoint Info/{tipo}
    """
    tipo = tipo.upper().strip()

    token_q = urllib.parse.quote(token, safe="")
    url = f"{BASE}/API/V1/Info/{tipo}?token={token_q}"

    r = requests.get(url, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"[SIAR] Error {r.status_code} en Info/{tipo}: {r.text[:500]}")
    return r.json()


def extract_info_raw(token: str, tipos: List[str], timeout: int = 30) -> Dict[str, dict]:
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
    Bronze histórico:
      - escribe un fichero con run_id por tipo (append)
      - opcional: escribe/actualiza latest_* para facilitar lectura rápida
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
    Carga los latest_* generados por run_info_bronze (si write_latest_copy=True)
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
    datos = payload.get("datos", [])
    if not isinstance(datos, list):
        raise ValueError("El payload no tiene 'datos' como lista.")
    return pd.DataFrame(datos)


def run_info_silver() -> Dict[str, Path]:
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