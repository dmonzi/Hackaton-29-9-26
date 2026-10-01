"""Mide el modelo: tableros sintéticos no usados al entrenar y los 10 diagramas de ONCE.

    python -m src.evaluar --modelo modelos/casillas.pt
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path
import cv2
from src.localizar import localizar_tablero
from src.nucleo import CLASES, cortar_casillas, etiquetas_a_fen, fen_a_etiquetas, leer_imagen
from src.orientacion import enderezar, esta_girado
from src.preparar_datos import es_fen, rutas_imagenes

def evaluar_sinteticos(clasificador, carpeta: str, n: int, excluir: set[str]) -> dict:
    rutas = [r for r in rutas_imagenes(carpeta) if es_fen(r) and os.path.basename(r) not in excluir]
    random.Random(1).shuffle(rutas)
    ok_casillas = exactos = total = 0
    for ruta in rutas[:n]:
        imagen = leer_imagen(ruta)
        if imagen is None:
            continue
        esperadas = fen_a_etiquetas(os.path.basename(ruta))
        vistas, _ = clasificador(imagen)  # la imagen ya es el tablero entero
        aciertos = sum(a == b for a, b in zip(vistas, esperadas))
        ok_casillas += aciertos
        exactos += aciertos == 64
        total += 1
    if not total:
        return {"tableros": 0}
    return {"tableros": total, "acierto_casilla": round(ok_casillas / (64 * total), 4),
            "tableros_exactos": round(exactos / total, 4)}

def evaluar_once(clasificador, carpeta: str = "tests/fixtures", solo: set[str] | None = None) -> list[dict]:
    res = []
    for fila in csv.DictReader(open(Path(carpeta) / "ejemplos_once.csv", encoding="utf-8")):
        if solo and fila["archivo"] not in solo:
            continue
        tablero = localizar_tablero(leer_imagen(str(Path(carpeta) / "img" / fila["archivo"])))
        vistas, confianzas = clasificador(tablero)
        girado = esta_girado(vistas)
        etiquetas = enderezar(vistas) if girado else vistas
        if girado:
            confianzas = confianzas[::-1]
        esperadas = fen_a_etiquetas(fila["fen"] + ".x")
        fallos = [f"{'abcdefgh'[i % 8]}{8 - i // 8}: esperaba {CLASES[e]}, vio {CLASES[v]} "
                  f"(confianza {confianzas[i]:.2f}{', se diría como duda' if confianzas[i] < 0.8 else ''})"
                  for i, (e, v) in enumerate(zip(esperadas, etiquetas)) if e != v]
        res.append({"archivo": fila["archivo"], "exacto": not fallos, "orientacion_ok": girado == (fila["girado"] == "1"),
                    "fen": etiquetas_a_fen(etiquetas), "fallos": fallos})
    return res

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="modelos/casillas.pt")
    p.add_argument("--carpeta", default="datos")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--usados", default="datos/casillas_tableros.txt", help="lista de tableros usados al entrenar")
    p.add_argument("--solo", nargs="*", help="evaluar solo estos ejemplos de ONCE (p. ej., los reservados)")
    a = p.parse_args()
    from src.reconocer import cargar_modelo, clasificar
    red = cargar_modelo(a.modelo)
    clasificador = lambda t: clasificar(t, red)
    excluir = set(Path(a.usados).read_text(encoding="utf-8").split()) if os.path.exists(a.usados) else set()
    sint = evaluar_sinteticos(clasificador, a.carpeta, a.n, excluir)
    once = evaluar_once(clasificador, solo=set(a.solo) if a.solo else None)
    print(f"Sintéticos no usados: {sint}")
    print(f"Ejemplos de ONCE exactos: {sum(r['exacto'] for r in once)} de {len(once)}")
    for r in once:
        print(f"  {r['archivo']}: {'exacto' if r['exacto'] else str(len(r['fallos'])) + ' casillas mal'}"
              f"{'' if r['orientacion_ok'] else ' (orientación mal)'}")
        for f in r["fallos"][:5]:
            print(f"      {f}")
    ruta = Path(a.modelo).parent / "metricas.json"
    metricas = json.load(open(ruta, encoding="utf-8")) if ruta.exists() else {}
    metricas[a.modelo] = {"sinteticos": sint, "once": [{k: r[k] for k in ("archivo", "exacto", "fallos")} for r in once]}
    json.dump(metricas, open(ruta, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Guardado en {ruta}")

if __name__ == "__main__":
    main()
