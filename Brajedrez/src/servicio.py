"""Lógica de la web: de archivos subidos (imágenes, PDF, Word) a una lista de diagramas analizados."""
import base64
import csv
import io
import re
import zipfile
from pathlib import Path
import cv2
import numpy as np
from src.localizar import localizar_tablero, localizar_tableros
from src.nucleo import leer_imagen
from src.pipeline import Clasificador, analizar_tablero

EXT_IMAGEN = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

def _decodificar(datos: bytes):
    return cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)

def imagenes_de_docx(datos: bytes) -> list[tuple[str, np.ndarray]]:
    """Imágenes de un .docx en el orden en que aparecen en el documento."""
    with zipfile.ZipFile(io.BytesIO(datos)) as z:
        rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
        destino = dict(re.findall(r'Id="([^"]+)"[^>]*Target="media/([^"]+)"', rels))
        destino.update({k: v for v, k in re.findall(r'Target="media/([^"]+)"[^>]*Id="([^"]+)"', rels)})
        orden = re.findall(r'r:embed="([^"]+)"', z.read("word/document.xml").decode("utf-8"))
        res = []
        for i, rid in enumerate([r for r in orden if r in destino], 1):
            img = _decodificar(z.read("word/media/" + destino[rid]))
            if img is not None:
                res.append((f"imagen {i}", img))
        return res

def paginas_de_pdf(datos: bytes, escala: float = 2.0) -> list[tuple[str, np.ndarray]]:
    import pypdfium2 as pdfium  # pip install pypdfium2
    pdf = pdfium.PdfDocument(datos)
    res = []
    for n in range(len(pdf)):
        rgb = np.array(pdf[n].render(scale=escala).to_pil().convert("RGB"))
        res.append((f"página {n + 1}", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
    return res

def miniatura(tablero: np.ndarray, lado: int = 240) -> str:
    ok, png = cv2.imencode(".png", cv2.resize(tablero, (lado, lado), interpolation=cv2.INTER_AREA))
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()

class Ejemplos:
    """Modo prueba sin modelo: reconoce los 10 diagramas de ONCE y usa su posición conocida."""
    def __init__(self, carpeta: str = "tests/fixtures"):
        self.filas = []
        for f in csv.DictReader(open(Path(carpeta) / "ejemplos_once.csv", encoding="utf-8")):
            t = localizar_tablero(leer_imagen(str(Path(carpeta) / "img" / f["archivo"])))
            self.filas.append((self._huella(t), f))

    @staticmethod
    def _huella(tablero):
        return cv2.resize(cv2.cvtColor(tablero, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_AREA).astype(float)

    def buscar(self, tablero):
        h = self._huella(tablero)
        dif, fila = min(((np.abs(h - hh).mean(), f) for hh, f in self.filas), key=lambda x: x[0])
        return fila if dif < 12 else None

def analizar_archivos(archivos: list[tuple[str, bytes]], clasificador: Clasificador | None = None,
                      ejemplos: Ejemplos | None = None) -> dict:
    diagramas, avisos = [], []
    for nombre, datos in archivos:
        ext = Path(nombre).suffix.lower()
        try:
            if ext == ".docx":
                fuentes = [(f"{nombre}, {e}", img, False) for e, img in imagenes_de_docx(datos)]
            elif ext == ".pdf":
                fuentes = [(f"{nombre}, {e}", img, True) for e, img in paginas_de_pdf(datos)]
            elif ext in EXT_IMAGEN or _decodificar(datos) is not None:
                fuentes = [(nombre, _decodificar(datos), False)]
            else:
                avisos.append(f"{nombre}: formato no admitido. Sube imágenes, PDF o Word (.docx).")
                continue
        except ImportError:
            avisos.append(f"{nombre}: para leer PDF hace falta instalar pypdfium2.")
            continue
        except Exception as e:  # archivo dañado
            avisos.append(f"{nombre}: no se ha podido abrir ({type(e).__name__}).")
            continue
        encontrados = 0
        for origen, img, es_pagina in fuentes:
            if img is None:
                continue
            tableros = localizar_tableros(img) if es_pagina else [t for t in [localizar_tablero(img)] if t is not None]
            for k, tablero in enumerate(tableros, 1):
                encontrados += 1
                sufijo = f", diagrama {k}" if len(tableros) > 1 else ""
                fila = ejemplos.buscar(tablero) if ejemplos else None
                if fila:
                    r = analizar_tablero(tablero, fen=fila["fen"], girado=fila["girado"] == "1")
                    r["aviso"] = "Modo prueba: posición tomada de los ejemplos de ONCE, no del modelo."
                else:
                    r = analizar_tablero(tablero, clasificador)
                r.update({"id": len(diagramas) + 1, "origen": origen + sufijo, "miniatura": miniatura(tablero)})
                diagramas.append(r)
        if not encontrados:
            avisos.append(f"{nombre}: no encuentro ningún tablero.")
    return {"diagramas": diagramas, "avisos": avisos}
