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
    CHECKPOINT_SIAR_DIARIOS_ESTACION,
    LOGS_DIR,
)

import pandas as pd
import requests

BASE = SIAR_BASE_URL
BRONZE_DIR = DATA_BRONZE_DATOS_DIR
SILVER_DIR = DATA_SILVER_DIR
CHECKPOINT_PATH = CHECKPOINT_SIAR_DIARIOS_ESTACION
LOG_DIR = LOGS_DIR

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


def get_accesos(token: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Consulta el endpoint Info/ACCESOS de SIAR para obtener límites y consumo.

    Esta función devuelve el primer elemento del bloque 'datos' de la respuesta,
    que incluye métricas como accesos máximos por día, accesos consumidos,
    registros máximos y registros acumulados.

    Args:
        token (str): Token de autenticación para la API SIAR.
        timeout (int, optional): Tiempo máximo de espera de la petición HTTP.

    Returns:
        Dict[str, Any]: Diccionario con la información de cuotas y consumo.
    """
    url = f"{BASE}/API/V1/Info/ACCESOS"
    r = requests.get(url, params={"token": token}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    datos = payload.get("datos", [])
    return datos[0] if isinstance(datos, list) and datos else {}


def remaining_budget(accesos: Dict[str, Any]) -> Tuple[int, int]:
    """
    Calcula el presupuesto restante de accesos y registros del día.

    Args:
        accesos (Dict[str, Any]): Diccionario devuelto por Info/ACCESOS.

    Returns:
        Tuple[int, int]: Tupla con:
            - accesos_restantes_dia
            - registros_restantes_dia
    """
    max_acc = int(accesos.get("MaxAccesosDia", 0) or 0)
    used_acc = int(accesos.get("NumAccesosDiaActual", 0) or 0)
    max_reg = int(accesos.get("MaxRegistrosDia", 0) or 0)
    used_reg = int(accesos.get("RegistrosAcumuladosDia", 0) or 0)
    return max(0, max_acc - used_acc), max(0, max_reg - used_reg)


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

    # SIAR no permite fechas anteriores a 1999-01-01
    if d_ini < MIN_API_DATE:
        d_ini = MIN_API_DATE

    # SIAR no permite fechas futuras; usamos hoy como máximo duro
    today = date.today()
    if d_fin > today:
        d_fin = today

    if d_ini > d_fin:
        raise ValueError(
            f"Intervalo inválido tras saneado: fecha_ini={d_ini.isoformat()} fecha_fin={d_fin.isoformat()}"
        )

    return d_ini.isoformat(), d_fin.isoformat()


def fetch_diarios_estacion(token: str, estacion_id: str, fecha_ini: str, fecha_fin: str, datos_calculados: bool, timeout: int = DEFAULT_TIMEOUT, max_retries: int = 6,) -> Dict[str, Any]:
    """
    Descarga datos diarios de una estación SIAR para un intervalo de fechas.

    La función llama al endpoint /API/V1/Datos/Diarios/ESTACION, sanea antes
    las fechas de consulta y reintenta automáticamente si detecta el límite
    de peticiones por minuto.

    Args:
        token (str): Token de autenticación para la API SIAR.
        estacion_id (str): Identificador de la estación.
        fecha_ini (str): Fecha inicial en formato YYYY-MM-DD.
        fecha_fin (str): Fecha final en formato YYYY-MM-DD.
        datos_calculados (bool): Indica si se solicitan datos calculados.
        timeout (int, optional): Tiempo máximo de espera de la petición HTTP.
        max_retries (int, optional): Número máximo de reintentos ante rate limit.

    Returns:
        Dict[str, Any]: Respuesta JSON de la API convertida a diccionario.

    Raises:
        RuntimeError: Si la API devuelve error, si se supera el número de
            reintentos o si se detecta una restricción no recuperable.
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
            raise RuntimeError(
                f"[SIAR] Error {r.status_code} est={estacion_id} "
                f"fecha_ini={fecha_ini} fecha_fin={fecha_fin} "
                f"url={r.url} body={r.text[:500]}"
            )

        return r.json()

    raise RuntimeError("[SIAR] Excedidos reintentos por rate limit.")


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


def save_bronze_chunk(payload: Dict[str, Any], estacion_id: str, fecha_ini: str, fecha_fin: str, datos_calculados: bool,) -> Path:
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

    # eliminar columnas completamente vacías (causa del FutureWarning)
    df_new = df_new.dropna(axis=1, how="all")

    if df_new.empty:
        return

    if path.exists():
        df_old = pd.read_parquet(path)

        # aseguramos mismas columnas
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


def is_month_complete_in_silver(df_silver: Optional[pd.DataFrame], estacion: str, month_start: date, month_end: date) -> bool:
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
    station_bajas: Optional[Dict[str, Optional[str]]] = None,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    datos_calculados: bool = False,
    min_access_buffer: int = 5,
    min_records_buffer: int = 2000,
    sleep_s: float = 0.3,
) -> None:
    """
    Ejecuta la carga incremental de datos diarios SIAR por estación.

    Flujo general:
    - carga el checkpoint previo,
    - calcula el rango pendiente por estación,
    - divide el periodo en bloques mensuales,
    - evita llamar a SIAR si el mes ya está completo en Silver,
    - controla cuotas diarias mediante Info/ACCESOS,
    - descarga cada bloque en Bronze,
    - transforma el payload y actualiza Silver,
    - guarda el checkpoint tras cada bloque correcto.

    Args:
        token (str): Token de autenticación para la API SIAR.
        estaciones (List[str]): Lista de identificadores de estación.
        station_bajas (Optional[Dict[str, Optional[str]]], optional):
            Diccionario estación -> fecha_baja en formato ISO. Si una estación
            está dada de baja antes del inicio global, se omite.
        start_date (str, optional): Fecha inicial global en formato YYYY-MM-DD.
        end_date (Optional[str], optional): Fecha final global. Si no se indica,
            se usa ayer.
        datos_calculados (bool, optional): Indica si se descargan datos calculados.
        min_access_buffer (int, optional): Umbral mínimo de accesos restantes
            para seguir procesando.
        min_records_buffer (int, optional): Umbral mínimo de registros restantes
            para seguir procesando.
        sleep_s (float, optional): Pausa entre peticiones correctas.

    Returns:
        None
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
                fecha_ini, fecha_fin = sanitize_query_dates(fecha_ini, fecha_fin)
            except ValueError as ve:
                log_line(f"[SKIP] est={est} rango_original={a.isoformat()}..{b.isoformat()} motivo={repr(ve)}")
                continue

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