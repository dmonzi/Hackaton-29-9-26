# Brajedrez

Cada diagrama de ajedrez, en voz y en braille.

## Qué es
Brajedrez lee un diagrama de ajedrez (una captura de pantalla, un escaneo de libro, un PDF o un Word) y lo cuenta a una persona ciega: dónde está cada pieza, qué casillas están resaltadas y qué flechas hay. Lo entrega como texto para lector de pantalla y voz, y como transcripción braille en el formato de ONCE, con la signografía del Documento técnico B8.

## Por qué
Los diagramas de ajedrez son imágenes: quien no ve no puede leerlos ni con lector de pantalla. Los descriptores genéricos de imágenes tampoco sirven, porque se equivocan al ubicar piezas y, peor, no avisan cuando dudan. En un tablero, una pieza inventada cambia la posición entera.

## Objetivo
Que una persona ciega abra un diagrama, lo escuche o lo lea en braille y pueda fiarse de lo que oye. Si la herramienta duda de una casilla, lo dice.

## Qué lo hace innovador
- No describe la imagen: reconstruye la posición (FEN) con un clasificador de casillas entrenado por nosotros y genera las salidas con reglas. No puede inventar piezas.
- Habla el formato de ONCE: el audio y el braille salen idénticos a sus ejemplos oficiales, comprobado con tests.
- Lee también las casillas resaltadas y las flechas, que en los diagramas explican la idea de la jugada.
- Avisa de lo que no sabe: «No estoy seguro de la casilla D5».
- Funciona con capturas y con diagramas de libro rayados, endereza los vistos desde las negras y acepta PDF y Word con varios diagramas.

## Cómo se usa
Requiere Python 3.10 o superior.

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    python -m src.web

Abre http://localhost:8000, sube tus diagramas y pulsa «Analizar diagramas». La web usa el modelo final `modelos/casillas_v2.pt`. Para abrirla desde el móvil: `python -m src.web --host 0.0.0.0` y entra en http://IP-del-portátil:8000.

También hay línea de comandos:

    python -m src.cli describir tests/fixtures/img/diagrama23.png

Y pruebas automáticas (40 tests, incluidos los formatos de audio y braille de ONCE):

    python -m pytest -q

## Cómo funciona
Recorte del tablero, 64 casillas, clasificador CNN de 13 clases, orientación, resaltados y flechas, validación con reglas del ajedrez, y salidas de audio y braille. El modelo se entrena con tableros sintéticos de Kaggle y sus copias en estilo libro, y se ajusta con diagramas reales de ONCE.

## Resultados
- 99,9 % de casillas acertadas en 500 tableros sintéticos no vistos.
- Diagramas de ONCE reservados (nunca vistos por el modelo): las dos capturas, exactas; en el diagrama de libro falla 1 casilla de 64 y la herramienta avisa de que duda de ella.
- Braille idéntico al de ONCE en sus 6 tableros de ejemplo.

Detalle y cómo reproducirlo: [docs/evidencias/reserva.md](docs/evidencias/reserva.md).

## Estructura

| Carpeta | Contenido |
| --- | --- |
| `src/` | Código: visión, clasificador, formatos de audio y braille, web y CLI |
| `src/static/` | Interfaz web accesible |
| `modelos/` | `casillas_v2.pt` (final), `casillas.pt` (base), `casillas_medida.pt` (para medir con la reserva) y sus métricas |
| `tests/` | Pruebas y los 10 diagramas de ejemplo de ONCE |
| `docs/` | Detalle técnico de cada módulo, evidencias y material del reto |
| `datos/` | Solo para reentrenar, no se entrega: dataset de Kaggle en `datos/train` y casillas preparadas (`.npz`) |

Cómo reentrenar desde cero: [docs/LEEME_tecnico.md](docs/LEEME_tecnico.md).

## Hecho con Kiro
Specs (lector-diagramas y experiencia-accesible), steering con reglas numeradas, hooks, un skill y un agente de revisión de accesibilidad.