import random
import pytest
from src.formatos import a_audio, a_braille, braille_a_fen

# Copiados tal cual del documento de ONCE (variante sin «P» en los peones)
EJEMPLOS_BRAILLE = {
    1: ("Re⠂ Dd⠂ Ta⠂ Th⠂ Cc⠒ Cf⠒ Ac⠲ Ag⠒ a⠆ b⠆ c⠆ d⠒ e⠲ f⠆ g⠆ h⠆", "Re⠦ Dd⠦ Ta⠦ Th⠦ Cc⠖ Cf⠖ Ac⠢ Ac⠦ a⠶ b⠶ c⠶ d⠖ e⠢ f⠶ g⠢ h⠖"),
    2: ("Re⠂ Dd⠂ Ta⠂ Th⠂ Cc⠒ Af⠂ Af⠲ a⠆ b⠆ c⠆ e⠲ f⠆ g⠆ h⠆", "Re⠦ Dd⠦ Ta⠦ Th⠦ Cg⠲ Ac⠦ Af⠦ a⠶ b⠶ d⠖ e⠢ f⠶ g⠶ h⠶"),
    3: ("Re⠂ Dd⠂ Ta⠂ Th⠂ Cb⠂ Cf⠒ Af⠂ Ag⠢ a⠆ b⠆ c⠲ d⠲ e⠆ f⠆ g⠆ h⠆", "Re⠦ Dd⠦ Ta⠦ Th⠦ Cb⠦ Cf⠖ Ac⠦ Af⠦ a⠶ b⠶ c⠶ d⠶ e⠖ f⠢ g⠶ h⠖"),
    4: ("Rh⠂ a⠲ b⠒ c⠆ f⠒ g⠲ h⠒", "Ra⠦ a⠶ b⠖ f⠲ g⠢ h⠲"),
    5: ("Rg⠂ Dd⠒ Tc⠂ Cd⠆ Ch⠢ Ac⠒ c⠲ d⠢ e⠲ g⠆ h⠒", "Rg⠦ Db⠦ Ta⠆ Tb⠖ Ad⠖ Ad⠶ c⠢ e⠢ f⠖ f⠶ h⠖"),
    6: ("Rg⠂ Dd⠂ Ta⠂ Te⠂ Cd⠆ Cf⠒ Ac⠂ Ac⠆ b⠲ c⠒ d⠢ e⠲ f⠆ g⠆ h⠒", "Rg⠦ Dc⠶ Tb⠦ Tf⠦ Cb⠶ Cf⠖ Ac⠦ Ae⠶ b⠢ c⠢ d⠖ e⠢ f⠶ g⠶ h⠶"),
}

@pytest.mark.parametrize("n", EJEMPLOS_BRAILLE)
def test_braille_identico_a_once(n):
    blancas, negras = EJEMPLOS_BRAILLE[n]
    lineas = a_braille(braille_a_fen(blancas, negras)).split("\n")
    assert lineas[1] == f"Blancas: {blancas}"
    assert lineas[2] == f"Negras: {negras}"

def test_braille_marcas_tablero5():
    salida = a_braille(braille_a_fen(*EJEMPLOS_BRAILLE[5]), resaltadas={"amarillo": ["g3", "h5"], "rojo": ["f6"]},
                       flechas=[("c1", "f1"), ("d3", "g3"), ("h5", "f6")])
    assert ")Resaltadas en amarillo las casillas g⠒ y h⠢ y, en rojo, la casilla f⠖" in salida
    assert "h⠢¬⠒⠕¬f⠖(" in salida

def test_audio_identico_diagrama23():
    esperado = ("Blancas: Rey en H3, Dama en F5, Torre en F1, Caballo en E2, Peones en A6, C4, D5, G3 y G4.\n"
                "Negras: Rey en H7, Dama en G6, Torre en E7, Alfil en E3, Peones en A7, C5, E4, G5 y H6.")
    assert a_audio("8/p3r2k/P5qp/2pP1Qp1/2P1p1P1/4b1PK/4N3/5R2") == esperado

def test_audio_sin_palabras_prohibidas():
    texto = a_audio("r1bqkb1r/pp3ppp/3p4/4p3/4PBn1/2N5/PPP2PPP/R2QKB1R")
    assert "reina" not in texto.lower() and "⠂" not in texto

def test_ida_y_vuelta_braille():
    random.seed(1)
    for _ in range(200):
        casillas = [""] * 64
        elegidas = random.sample(range(64), random.randint(2, 32))
        casillas[elegidas[0]], casillas[elegidas[1]] = "K", "k"
        for s in elegidas[2:]:
            casillas[s] = random.choice("QRBNPqrbnp")
        filas = []
        for r in range(8):
            fila, v = "", 0
            for x in casillas[r * 8:r * 8 + 8]:
                if x:
                    fila += (str(v) if v else "") + x
                    v = 0
                else:
                    v += 1
            filas.append(fila + (str(v) if v else ""))
        fen = "/".join(filas)
        lineas = a_braille(fen).split("\n")
        assert braille_a_fen(lineas[1].split(": ", 1)[1], lineas[2].split(": ", 1)[1]) == fen

def test_validacion():
    from src.validar import problemas
    assert problemas("8/p3r2k/P5qp/2pP1Qp1/2P1p1P1/4b1PK/4N3/5R2") == []
    assert problemas("K7/8/8/8/8/8/8/K6k")          # dos reyes blancos
    assert problemas("P6k/8/8/8/8/8/8/K7")          # peón en la fila 8
