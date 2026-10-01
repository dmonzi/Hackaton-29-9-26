"""Casillas resaltadas y flechas en diagramas digitales (estilo chess.com: flechas naranjas)."""
import cv2
import numpy as np

DIFERENCIA_MIN = 30  # distancia Lab al color normal de la casilla para que un píxel naranja cuente como flecha

def _nombre(fila: int, col: int, girado: bool) -> str:
    """fila/col de pantalla (0 = arriba/izquierda) -> casilla algebraica."""
    return ("abcdefgh"[7 - col] + str(fila + 1)) if girado else ("abcdefgh"[col] + str(8 - fila))

def casillas_resaltadas(tablero: np.ndarray, girado: bool = False) -> dict:
    """Compara el color de las esquinas libres de cada casilla con el color normal de su tipo (clara/oscura)."""
    lado = tablero.shape[0] // 8
    m = max(3, lado // 6)
    lab = cv2.cvtColor(tablero, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(tablero, cv2.COLOR_BGR2HSV)
    muestras, tonos = {}, {}
    for f in range(8):
        for c in range(8):
            y0, x0 = f * lado, c * lado
            # esquina superior derecha e inferior izquierda: sin coordenadas impresas ni pieza
            zonas = [(slice(y0 + 2, y0 + 2 + m), slice(x0 + lado - 2 - m, x0 + lado - 2)),
                     (slice(y0 + lado - 2 - m, y0 + lado - 2), slice(x0 + 2, x0 + 2 + m))]
            muestras[f, c] = np.median(np.vstack([lab[z].reshape(-1, 3) for z in zonas]), axis=0)
            tonos[f, c] = np.median(np.vstack([hsv[z].reshape(-1, 3) for z in zonas]), axis=0)
    base = {p: np.median([v for (f, c), v in muestras.items() if (f + c) % 2 == p], axis=0) for p in (0, 1)}
    res = {}
    for (f, c), v in muestras.items():
        if np.linalg.norm(v - base[(f + c) % 2]) < 25:
            continue
        h, s, _ = tonos[f, c]
        color = "rojo" if (h < 12 or h > 165) and s > 80 else "amarillo" if 18 <= h <= 45 else None
        if color:
            res.setdefault(color, []).append(_nombre(f, c, girado))
    return {k: sorted(res[k]) for k in ("amarillo", "rojo") if k in res}  # mismo orden que ONCE

def flechas(tablero: np.ndarray, girado: bool = False) -> list[tuple[str, str]]:
    """Flechas naranjas -> [(origen, destino)]. La punta (horquilla del esqueleto) marca el destino."""
    from skimage.morphology import skeletonize
    lado = tablero.shape[0] / 8
    hsv = cv2.cvtColor(tablero, cv2.COLOR_BGR2HSV)
    naranja = cv2.inRange(hsv, (8, 120, 140), (26, 255, 255))
    # en tableros marrones o de madera las casillas ya son naranjas: solo cuenta lo que se aparta de su color normal
    lab = cv2.cvtColor(tablero, cv2.COLOR_BGR2LAB).astype(np.float32)
    filas, cols = np.indices(tablero.shape[:2])
    paridad = ((filas // lado).astype(int) + (cols // lado).astype(int)) % 2
    base = np.stack([np.median(lab[paridad == q], axis=0) for q in (0, 1)])
    distinto = np.linalg.norm(lab - base[paridad], axis=2) > DIFERENCIA_MIN
    mascara = cv2.morphologyEx(naranja & (distinto.astype(np.uint8) * 255),
                               cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara)
    casilla = lambda p: _nombre(int(np.clip(p[1] // lado, 0, 7)), int(np.clip(p[0] // lado, 0, 7)), girado)
    resultado = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 0.1 * lado * lado:
            continue
        comp = etiquetas == k
        pts = np.argwhere(comp)[:, ::-1].astype(float)
        # una pieza dorada cabe en su casilla; una flecha une dos casillas, así que ocupa al menos dos
        _, por_casilla = np.unique((pts[:, 1] // lado) * 8 + pts[:, 0] // lado, return_counts=True)
        if (por_casilla >= 0.15 * len(pts)).sum() < 2:
            continue
        if max(stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]) < 1.3 * lado:
            # flecha corta (casilla vecina): la punta es el extremo más ancho a lo largo del eje
            centro = pts.mean(axis=0)
            _, _, ejes = np.linalg.svd(pts - centro, full_matrices=False)
            t, lat = (pts - centro) @ ejes[0], (pts - centro) @ ejes[1]
            tercio = (t.max() - t.min()) / 3
            if np.ptp(lat[t > t.max() - tercio]) >= np.ptp(lat[t < t.min() + tercio]):
                punta, cola, sentido = centro + t.max() * ejes[0], centro + t.min() * ejes[0], ejes[0]
            else:
                punta, cola, sentido = centro + t.min() * ejes[0], centro + t.max() * ejes[0], -ejes[0]
            destino = casilla(punta - 0.15 * lado * sentido)
            for paso in (0.35, 0.6, 0.85):
                origen = casilla(cola - paso * lado * sentido)
                if origen != destino:
                    break
            resultado.append((origen, destino))
            continue
        esq = skeletonize(comp).astype(np.uint8)
        vecinos = cv2.filter2D(esq, -1, np.ones((3, 3), np.float32)) - 1
        extremos = np.argwhere((esq == 1) & (vecinos == 1))[:, ::-1].astype(float)  # (x, y)
        # agrupar extremos cercanos: grupo de 2 o más = punta (horquilla); grupo de 1 = cola
        grupos = []
        for p in extremos:
            for g in grupos:
                if np.linalg.norm(g[0] - p) < 0.45 * lado:
                    g.append(p)
                    break
            else:
                grupos.append([p])
        puntas = [np.mean(g, axis=0) for g in grupos if len(g) >= 2]
        colas = [g[0] for g in grupos if len(g) == 1]
        for punta in puntas:
            if colas:
                cola = min(colas, key=lambda c: np.linalg.norm(c - punta)) if len(colas) > 1 else colas[0]
            else:  # sin cola visible: el punto del trazo más lejano a la punta
                cola = pts[np.argmax(np.linalg.norm(pts - punta, axis=1))]
            cerca = pts[np.linalg.norm(pts - cola, axis=1) < 0.6 * lado]
            hacia = cerca.mean(axis=0) - cola
            unidad = hacia / (np.linalg.norm(hacia) + 1e-6)
            for paso in (0.35, 0.6, 0.85):  # una flecha nunca empieza y acaba en la misma casilla
                origen = casilla(cola - paso * lado * unidad)
                if origen != casilla(punta):
                    break
            resultado.append((origen, casilla(punta)))
    return sorted(set(resultado))
