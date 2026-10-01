import numpy as np

def estilo_libro(casilla: np.ndarray, oscura: bool, paso: int = 4) -> np.ndarray:
    """Convierte una casilla en gris (estilo digital) al estilo de libro impreso:
    fondo blanco o rayado en diagonal y piezas en blanco y negro."""
    lado = casilla.shape[0]
    d = max(2, lado // 12)  # anillo interior: evita el borde de la casilla vecina
    anillo = casilla[d:-d, d:-d]
    borde = np.concatenate([anillo[0], anillo[-1], anillo[:, 0], anillo[:, -1]])
    fondo = np.abs(casilla.astype(int) - int(np.median(borde))) < 14
    pieza = np.where(casilla < 110, 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:lado, 0:lado]
    rayas = np.where((xx + yy) % paso == 0, 0, 255).astype(np.uint8) if oscura else np.full((lado, lado), 255, np.uint8)
    return np.where(fondo, rayas, pieza).astype(np.uint8)
