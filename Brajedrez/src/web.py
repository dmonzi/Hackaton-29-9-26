"""Servidor de la web, sin dependencias extra (biblioteca estándar de Python).

    python -m src.web                      # http://127.0.0.1:8000, usa modelos/casillas_v2.pt si existe
    python -m src.web --host 0.0.0.0       # para abrirla desde el móvil en la misma red
    python -m src.web --prueba             # sin modelo: reconoce solo los 10 diagramas de ejemplo de ONCE
"""
import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from src.servicio import Ejemplos, analizar_archivos

ESTATICO = Path(__file__).parent / "static"
MAX_BYTES = 40 * 1024 * 1024
# Configuración del servidor. Va en un diccionario y no como atributo de la clase Manejador:
# una función guardada en la clase se convertiría en método y recibiría «self» como primer argumento.
CONFIG = {"clasificador": None, "ejemplos": None}

def cargar_clasificador(ruta: str):
    if not Path(ruta).exists():
        print(f"Aviso: no existe {ruta}; la web localizará tableros pero no podrá leer las piezas.")
        return None
    try:
        from src.reconocer import cargar_modelo, clasificar
    except ImportError:
        print("Aviso: PyTorch no está instalado; no se puede usar el modelo.")
        return None
    modelo = cargar_modelo(ruta)
    return lambda tablero: clasificar(tablero, modelo)

class Manejador(BaseHTTPRequestHandler):
    def _enviar(self, codigo: int, cuerpo: bytes, tipo: str) -> None:
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, codigo: int, datos: dict) -> None:
        self._enviar(codigo, json.dumps(datos, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._enviar(200, (ESTATICO / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/estado":
            self._json(200, {"modelo": CONFIG["clasificador"] is not None, "prueba": CONFIG["ejemplos"] is not None})
        else:
            self._json(404, {"error": "No encontrado."})

    def do_POST(self):
        if self.path != "/api/analizar":
            return self._json(404, {"error": "No encontrado."})
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BYTES:
            return self._json(413, {"error": "Los archivos pesan más de 40 MB en total. Súbelos en varias veces."})
        try:
            cuerpo = json.loads(self.rfile.read(n))
            archivos = [(a["nombre"], base64.b64decode(a["datos"])) for a in cuerpo["archivos"]]
        except (ValueError, KeyError, TypeError):
            return self._json(400, {"error": "La petición no tiene el formato esperado."})
        self._json(200, analizar_archivos(archivos, CONFIG["clasificador"], CONFIG["ejemplos"]))

def main() -> None:
    p = argparse.ArgumentParser(description="Web accesible para leer diagramas de ajedrez")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--puerto", type=int, default=8000)
    p.add_argument("--modelo", default="modelos/casillas_v2.pt")
    p.add_argument("--prueba", action="store_true", help="reconoce los 10 diagramas de ejemplo de ONCE sin modelo")
    a = p.parse_args()
    CONFIG["clasificador"] = cargar_clasificador(a.modelo)
    CONFIG["ejemplos"] = Ejemplos() if a.prueba else None
    print(f"Web lista en http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.puerto}")
    ThreadingHTTPServer((a.host, a.puerto), Manejador).serve_forever()

if __name__ == "__main__":
    main()
