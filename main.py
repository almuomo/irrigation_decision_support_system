
from scripts.get_token import get_siar_token
from scripts.etl_info_siar import run_info_pipeline
# from scripts.etl_datos_siar import run_datos_pipeline

def main():
    # Paso 1: obtener token (con cache en memoria)
    token = get_siar_token()

    # Paso 2: Ejecución pipeline info
    run_info_pipeline(token=token)

if __name__ == "__main__":
    main()