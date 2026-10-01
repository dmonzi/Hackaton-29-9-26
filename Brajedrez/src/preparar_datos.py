"""Prepara casillas etiquetadas desde el dataset chess-positions (FEN en el nombre del archivo).
Añade una copia en «estilo libro» (fondo rayado, blanco y negro) de parte de las casillas.

    python -m src.preparar_datos --carpeta datos --tableros 6000
"""
import argparse
import glob
import os
import random
from pathlib import Path
import cv2
import numpy as np
from src.estilo_libro import estilo_libro
from src.nucleo import cortar_casillas, fen_a_etiquetas, leer_imagen

def rutas_imagenes(carpeta: str) -> list[str]:
    return sorted(p for ext in ("jpeg", "jpg", "png")
                  for p in glob.glob(os.path.join(carpeta, "**", f"*.{ext}"), recursive=True))

def es_fen(ruta: str) -> bool:
    try:
        fen_a_etiquetas(os.path.basename(ruta))
        return True
    except Exception:
        return False

def lista_usados(salida_npz: str) -> str:
    return str(Path(salida_npz).with_suffix("")) + "_tableros.txt"

def preparar(carpeta: str = "datos", n_tableros: int = 6000, prop_vacias: float = 0.15, p_libro: float = 0.5,
             salida: str = "datos/casillas.npz") -> None:
    random.seed(0)
    rutas = [r for r in rutas_imagenes(carpeta) if es_fen(r)]
    if not rutas:
        raise SystemExit(f"No hay imágenes con la FEN en el nombre dentro de «{carpeta}».")
    random.shuffle(rutas)
    usadas = rutas[:n_tableros]
    X, y = [], []
    for k, ruta in enumerate(usadas, 1):
        imagen = leer_imagen(ruta)
        if imagen is None:
            continue
        for i, (casilla, etiqueta) in enumerate(zip(cortar_casillas(imagen), fen_a_etiquetas(os.path.basename(ruta)))):
            if etiqueta or random.random() < prop_vacias:
                X.append(casilla)
                y.append(etiqueta)
                if random.random() < p_libro:
                    X.append(estilo_libro(casilla, oscura=(i // 8 + i % 8) % 2 == 1))
                    y.append(etiqueta)
        if k % 1000 == 0:
            print(f"  {k} tableros procesados")
    np.savez_compressed(salida, X=np.array(X, np.uint8), y=np.array(y, np.int64))
    Path(lista_usados(salida)).write_text("\n".join(os.path.basename(r) for r in usadas), encoding="utf-8")
    print(f"{len(usadas)} tableros de {len(rutas)} disponibles -> {len(y)} casillas en {salida}")
    print(f"Reparto por clase (vacía, P N B R Q K, p n b r q k): {np.bincount(y, minlength=13).tolist()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--carpeta", default="datos")
    p.add_argument("--tableros", type=int, default=6000)
    p.add_argument("--p-libro", type=float, default=0.5)
    p.add_argument("--salida", default="datos/casillas.npz")
    a = p.parse_args()
    preparar(a.carpeta, a.tableros, p_libro=a.p_libro, salida=a.salida)
