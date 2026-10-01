import base64
import csv
import io
import json
import threading
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np
import pytest
from src.servicio import Ejemplos, analizar_archivos

FIX = Path(__file__).parent / "fixtures"
FILAS = list(csv.DictReader(open(FIX / "ejemplos_once.csv", encoding="utf-8")))
EJEMPLOS = Ejemplos(str(FIX))

def _bytes(nombre):
    return (FIX / "img" / nombre).read_bytes()

def _docx(nombres):
    """Word mínimo con las imágenes en orden (solo las partes que lee el extractor)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        rels = "".join(f'<Relationship Id="rId{i}" Type="image" Target="media/i{i}.png"/>' for i in range(len(nombres)))
        z.writestr("word/_rels/document.xml.rels", f"<Relationships>{rels}</Relationships>")
        z.writestr("word/document.xml", "<w:document>" + "".join(f'<a:blip r:embed="rId{i}"/>' for i in range(len(nombres))) + "</w:document>")
        for i, n in enumerate(nombres):
            z.writestr(f"word/media/i{i}.png", _bytes(n))
    return buf.getvalue()

def test_word_con_los_10_ejemplos():
    r = analizar_archivos([("ejemplos.docx", _docx([f["archivo"] for f in FILAS]))], ejemplos=EJEMPLOS)
    assert [d["fen"] for d in r["diagramas"]] == [f["fen"] for f in FILAS]
    assert r["diagramas"][4]["origen"] == "ejemplos.docx, imagen 5"

def test_pdf_con_dos_diagramas_en_una_pagina():
    pytest.importorskip("pypdfium2")
    from PIL import Image
    pagina = Image.new("RGB", (1240, 1754), "white")
    pagina.paste(Image.open(FIX / "img" / "diagrama1.png").resize((580, 514)), (330, 300))
    pagina.paste(Image.open(FIX / "img" / "diagrama23.png").resize((540, 460)), (350, 1100))
    buf = io.BytesIO()
    pagina.save(buf, "PDF", resolution=150)
    r = analizar_archivos([("libro.pdf", buf.getvalue())], ejemplos=EJEMPLOS)
    assert [d["origen"] for d in r["diagramas"]] == ["libro.pdf, página 1, diagrama 1", "libro.pdf, página 1, diagrama 2"]

def test_sin_modelo_avisa_y_no_inventa():
    d = analizar_archivos([("tablero5.png", _bytes("tablero5.png"))])["diagramas"][0]
    assert "Falta el modelo" in d["error"] and "audio" not in d

def test_casilla_dudosa_se_dice_al_final():
    from src.nucleo import fen_a_etiquetas
    fen = FILAS[4]["fen"]
    def clasificador(_):
        confianzas = [0.99] * 64
        confianzas[27] = 0.5  # d5
        return fen_a_etiquetas(fen + ".x"), confianzas
    d = analizar_archivos([("tablero5.png", _bytes("tablero5.png"))], clasificador)["diagramas"][0]
    assert d["audio"].endswith("No estoy seguro de la casilla D5.")

def test_avisos_de_formato_y_sin_tablero():
    blanca = cv2.imencode(".png", np.full((300, 300, 3), 255, np.uint8))[1].tobytes()
    r = analizar_archivos([("nota.txt", b"hola"), ("blanca.png", blanca)])
    assert r["diagramas"] == [] and len(r["avisos"]) == 2

def test_servidor_web():
    from src.web import CONFIG, Manejador
    CONFIG["clasificador"], CONFIG["ejemplos"] = None, EJEMPLOS
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), Manejador)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{servidor.server_address[1]}"
    try:
        assert "Lector de voz" in urllib.request.urlopen(url + "/").read().decode("utf-8")
        cuerpo = json.dumps({"archivos": [{"nombre": "t2.png", "datos": base64.b64encode(_bytes("tablero2.png")).decode()}]})
        pet = urllib.request.Request(url + "/api/analizar", cuerpo.encode(), {"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(pet).read())["diagramas"][0]
        assert d["girado"] is True and d["flechas"] == [["d4", "d1"]] and d["braille"].startswith("Tablero:")
    finally:
        servidor.shutdown()
