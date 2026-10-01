"""Uso: python -m src.cli describir <imagen> [--modelo modelos/casillas_v2.pt] [--json] [--foto]
Sin modelo todavía: --fen <colocación> [--girado] para probar marcas y formatos."""
import argparse
import json
import sys
from src.localizar import localizar_tablero
from src.nucleo import leer_imagen
from src.pipeline import analizar_tablero

def describir(ruta: str, modelo: str | None, fen: str | None, girado: bool, foto: bool) -> dict:
    imagen = leer_imagen(ruta)
    tablero = localizar_tablero(imagen, en_perspectiva=foto) if imagen is not None else None
    if tablero is None:
        return {"error": "No encuentro un tablero en la imagen."}
    if fen is not None:
        return analizar_tablero(tablero, fen=fen, girado=girado)
    from src.reconocer import cargar_modelo, clasificar
    red = cargar_modelo(modelo or "modelos/casillas_v2.pt")
    return analizar_tablero(tablero, lambda t: clasificar(t, red))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("orden", choices=["describir"])
    p.add_argument("imagen")
    p.add_argument("--modelo")
    p.add_argument("--fen")
    p.add_argument("--girado", action="store_true")
    p.add_argument("--foto", action="store_true", help="foto con el móvil: corrige la perspectiva")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # el braille no cabe en cp1252 (consola de Windows)
    r = describir(a.imagen, a.modelo, a.fen, a.girado, a.foto)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif "error" in r:
        print(r["error"])
        sys.exit(1)
    else:
        print(r["audio"] + "\n\n" + r["braille"])

if __name__ == "__main__":
    main()
