import os
import time
import urllib.parse
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://servicio.mapa.gob.es/siarapi"

# --- Cache en memoria ---
_TOKEN_CACHE = None
_TOKEN_TS = 0.0
TOKEN_TTL = 50 * 60  # 50 min


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}. Revisa tu .env")
    return value


def get_siar_token(base_url: str = BASE_URL, timeout: int = 30, force_refresh: bool = False) -> str:
    global _TOKEN_CACHE, _TOKEN_TS

    now = time.time()
    if (not force_refresh) and (_TOKEN_CACHE is not None) and ((now - _TOKEN_TS) < TOKEN_TTL):
        return _TOKEN_CACHE

    user = _require_env("SIAR_USER")
    password = _require_env("SIAR_PASS")

    # IMPORTANTE: replicamos tu comportamiento "simple": construir URL con querystring
    url_user = f"{base_url}/API/V1/Autenticacion/cifrarCadena?cadena={user}"
    url_password = f"{base_url}/API/V1/Autenticacion/cifrarCadena?cadena={password}"

    resp_user = requests.get(url_user, timeout=timeout)
    resp_user.raise_for_status()

    resp_pass = requests.get(url_password, timeout=timeout)
    resp_pass.raise_for_status()

    # IMPORTANTE: igual que tu "simple" (sin safe="")
    user_enc = urllib.parse.quote(resp_user.text.strip())
    pass_enc = urllib.parse.quote(resp_pass.text.strip())

    url_token = f"{base_url}/API/V1/Autenticacion/obtenerToken?Usuario={user_enc}&Password={pass_enc}"
    r_token = requests.get(url_token, timeout=timeout)
    r_token.raise_for_status()

    token = r_token.text.strip()
    if not token:
        raise RuntimeError("Token vacío. Revisa credenciales o respuesta del servicio.")

    _TOKEN_CACHE = token
    _TOKEN_TS = now
    return token


if __name__ == "__main__":
    token = get_siar_token(force_refresh=True)
    print("Token OK")
    print("len:", len(token))
    print("head:", token[:10])