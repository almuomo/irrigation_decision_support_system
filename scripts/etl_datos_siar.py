from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from calendar import monthrange

import pandas as pd
import requests

BASE = "https://servicio.mapa.gob.es/siarapi"

BRONZE_DIR = Path("data/bronce/datos")
SILVER_DIR = Path("data/silver")
CHECKPOINT_PATH = Path("./info/checkpoints/siar_diarios_estacion.json")
LOG_DIR = Path("./info/logs")

DEFAULT_TIMEOUT = 60


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_checkpoint() -> Dict[str, str]:
    if not CHECKPOINT_PATH.exists():
        return {}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(state: Dict[str, str]) -> None:
    ensure_dir(CHECKPOINT_PATH.parent)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)


def log_line(msg: str) -> None:
    """
    Log simple a fichero diario + consola.
    """
    ensure_dir(LOG_DIR)
    log_path = LOG_DIR / f"siar_diarios_{date.today().isoformat()}.log"
    line = f"{utc_now().isoformat()} | {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_accesos(token: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Llama a Info/ACCESOS para conocer límites y consumo.
    """
    url = f"{BASE}/API/V1/Info/ACCESOS"
    r = requests.get(url, params={"token": token}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    datos = payload.get("datos", [])
    return datos[0] if isinstance(datos, list) and datos else {}


def remaining_budget(accesos: Dict[str, Any]) -> Tuple[int, int]:
    """
    Devuelve (accesos_restantes_dia, registros_restantes_dia)
    """
    max_acc = int(accesos.get("MaxAccesosDia", 0) or 0)
    used_acc = int(accesos.get("NumAccesosDiaActual", 0) or 0)
    max_reg = int(accesos.get("MaxRegistrosDia", 0) or 0)
    used_reg = int(accesos.get("RegistrosAcumuladosDia", 0) or 0)
    return max(0, max_acc - used_acc), max(0, max_reg - used_reg)


def fetch_diarios_estacion(
    token: str,
    estacion_id: str,
    fecha_ini: str,
    fecha_fin: str,
    datos_calculados: bool,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 6,
) -> Dict[str, Any]:
    """
    GET /API/V1/Datos/Diarios/ESTACION
    Reintenta si hay rate limit (5/min).
    """
    url = f"{BASE}/API/V1/Datos/Diarios/ESTACION"
    params = {
        "token": token,
        "Id": estacion_id,
        "FechaInicial": fecha_ini,
        "FechaFinal": fecha_fin,
        "DatosCalculados": "true" if datos_calculados else "false",
    }

    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params, timeout=timeout)

        # Rate limit por minuto
        if r.status_code == 403 and "número de peticiones permitidas en un minuto" in r.text.lower():
            wait_s = 15  # margen sobre 12s
            print(f"[RATE] Límite por minuto. Esperando {wait_s}s (intento {attempt}/{max_retries})...")
            time.sleep(wait_s)
            continue

        # Fecha mínima autorizada (lo mantenemos informativo)
        if r.status_code == 403 and "fecha inicial" in r.text.lower() and "inferior" in r.text.lower():
            raise RuntimeError(f"[SIAR] 403 por fecha mínima: {r.text[:500]}")

        if not r.ok:
            raise RuntimeError(f"[SIAR] Error {r.status_code} {r.text[:500]}")

        return r.json()

    raise RuntimeError("[SIAR] Excedidos reintentos por rate limit.")


def payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
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
    ensure_dir(BRONZE_DIR)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    calc_tag = "calc1" if datos_calculados else "calc0"
    path = BRONZE_DIR / f"{ts}_diarios_estacion_{estacion_id}_{fecha_ini}_{fecha_fin}_{calc_tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def upsert_silver_append_dedupe(df_new: pd.DataFrame, path: Path, key_cols: List[str]) -> None:
    """
    Append + dedupe por claves. Mantiene el último registro en caso de colisión.
    """
    ensure_dir(path.parent)

    if path.exists():
        df_old = pd.read_parquet(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new.copy()

    keys = [c for c in key_cols if c in df_all.columns]
    if keys:
        df_all = df_all.drop_duplicates(subset=keys, keep="last")

    df_all.to_parquet(path, index=False)

def load_silver_minimal(silver_path: Path, cols: List[str]) -> Optional[pd.DataFrame]:
    """
    Carga solo columnas mínimas de Silver (para checks rápidos).
    """
    if not silver_path.exists():
        return None
    return pd.read_parquet(silver_path, columns=[c for c in cols if c])


def is_month_complete_in_silver(
    df_silver: Optional[pd.DataFrame],
    estacion: str,
    month_start: date,
    month_end: date
) -> bool:
    """
    True si en Silver ya existen datos para TODOS los días del rango [month_start, month_end]
    para esa estación.
    """
    if df_silver is None or df_silver.empty:
        return False

    if "Estacion" not in df_silver.columns or "Fecha" not in df_silver.columns:
        return False

    # días esperados (YYYY-MM-DD)
    expected = set(
        (month_start + timedelta(days=i)).isoformat()
        for i in range((month_end - month_start).days + 1)
    )

    # Fechas existentes para esa estación
    fechas = df_silver.loc[df_silver["Estacion"] == estacion, "Fecha"].astype(str)
    existing = set(fechas[fechas.isin(expected)].tolist())

    return expected.issubset(existing)

def month_chunks(start: date, end: date, step_months: int = 1) -> List[Tuple[date, date]]:
    """
    Devuelve rangos [ini, fin] agrupando step_months meses naturales.
    """
    out: List[Tuple[date, date]] = []
    cur = start.replace(day=1)

    while cur <= end:
        # calcular mes final del bloque
        year = cur.year
        month = cur.month + step_months - 1

        # ajustar overflow de meses
        while month > 12:
            month -= 12
            year += 1

        last_day = monthrange(year, month)[1]
        block_end = date(year, month, last_day)

        a = max(start, cur)
        b = min(end, block_end)
        out.append((a, b))

        # avanzar step_months meses
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
    station_bajas: Optional[Dict[str, Optional[str]]] = None,  # <-- NUEVO
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    datos_calculados: bool = False,
    min_access_buffer: int = 5,
    min_records_buffer: int = 2000,
    sleep_s: float = 0.3,
) -> None:
    """
    Descarga incremental por estación desde start_date hasta end_date (por defecto hoy-1).
    Para cada estación usa checkpoint y respeta cuotas (Info/ACCESOS).
    """
    state = load_checkpoint()
    

    if end_date is None:
        end_date = str(date.today() - timedelta(days=1))

    d_global_start = date.fromisoformat(start_date)
    d_end = date.fromisoformat(end_date)

    silver_path = SILVER_DIR / f"siar_diarios_estacion_calc{int(datos_calculados)}.parquet"

    df_silver_cache = load_silver_minimal(silver_path, cols=["Fecha", "Estacion"])

    key_cols = ["Fecha", "Estacion"]

    for est in estaciones:
        ck_key = f"Diarios|ESTACION|{est}|calc={str(datos_calculados).lower()}"

        # --- SKIP estación si está dada de baja antes del start_date global ---
        if station_bajas is not None:
            fb = station_bajas.get(est)  # "YYYY-MM-DD" o None
            if fb:
                fb_date = date.fromisoformat(fb)
                if fb_date < d_global_start:
                    # marcamos checkpoint al final para no reintentar nunca
                    state[ck_key] = d_end.isoformat()
                    save_checkpoint(state)
                    log_line(
                        f"[SKIP] est={est} motivo=fecha_baja<{d_global_start.isoformat()} "
                        f"fecha_baja={fb} -> no se extrae"
                    )
                    continue

        last = state.get(ck_key)
        if last:
            d_start = date.fromisoformat(last) + timedelta(days=1)
        else:
            d_start = d_global_start

        if d_start > d_end:
            continue

        chunks = month_chunks(d_start, d_end, step_months=1)

        for a, b in chunks:
            # Si ya tengo el mes completo en Silver, no llamo a SIAR (ahorra cuota)
            if is_month_complete_in_silver(df_silver_cache, est, a, b):
                state[ck_key] = b.isoformat()
                save_checkpoint(state)
                log_line(f"[SKIP] est={est} rango={a.isoformat()}..{b.isoformat()} motivo=month_complete")
                continue
            accesos = get_accesos(token)
            acc_left, reg_left = remaining_budget(accesos)

            if acc_left <= min_access_buffer or reg_left <= min_records_buffer:
                log_line(f"[STOP] quota_low accesos_left={acc_left} registros_left={reg_left}")
                save_checkpoint(state)
                return

            fecha_ini = a.isoformat()
            fecha_fin = b.isoformat()

            try:
                payload = fetch_diarios_estacion(
                    token=token,
                    estacion_id=est,
                    fecha_ini=fecha_ini,
                    fecha_fin=fecha_fin,
                    datos_calculados=datos_calculados,
                )

                bronze_path = save_bronze_chunk(payload, est, fecha_ini, fecha_fin, datos_calculados)

                df = payload_to_df(payload)
                if not df.empty:
                    df["ingestion_utc"] = utc_now().isoformat()
                    upsert_silver_append_dedupe(df, silver_path, key_cols)

                df_silver_cache = load_silver_minimal(silver_path, cols=["Fecha", "Estacion"])

                state[ck_key] = fecha_fin
                save_checkpoint(state)

                log_line(
                    f"[OK] est={est} rango={fecha_ini}..{fecha_fin} filas={len(df)} bronze={bronze_path.name} "
                    f"accesos_left={acc_left} registros_left={reg_left}"
                )

                time.sleep(sleep_s)

            except Exception as e:
                # guardas checkpoint hasta lo último que estuviera OK
                save_checkpoint(state)
                log_line(f"[ERROR] est={est} rango={fecha_ini}..{fecha_fin} err={repr(e)}")
                raise

    save_checkpoint(state)
    log_line("[DONE] Incremental completo.")