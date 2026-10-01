# Kit de referencia: lector accesible de diagramas (reto ONCE)

Se ejecuta todo desde la carpeta raíz del proyecto. Módulos probados con los 10 diagramas de ejemplo de ONCE, para usar desde las tareas de Kiro (regla R13).

| Fichero | Qué hace | Estado |
| --- | --- | --- |
| src/formatos.py | Audio y braille B8 en el formato de ONCE; braille -> FEN | Probado: braille idéntico 6/6, audio idéntico al diagrama 23 |
| src/localizar.py | Recorta el tablero (capturas y escaneos; `en_perspectiva=True` para fotos) | Probado 10/10 |
| src/marcas.py | Casillas resaltadas (amarillo, rojo) y flechas naranjas de chess.com | Probado 10/10 |
| src/orientacion.py | Detecta y endereza diagramas vistos desde las negras | Probado |
| src/validar.py | Posiciones imposibles (regla R3) | Probado |
| src/estilo_libro.py | Aumento: convierte casillas digitales a estilo libro rayado | Revisado a ojo |
| src/preparar_datos.py | Casillas del dataset + copias estilo libro -> .npz, y lista de tableros usados | Probado |
| src/evaluar.py | Acierto en sintéticos no usados y en los 10 ejemplos de ONCE -> modelos/metricas.json | Probado con un clasificador de sustitución |
| src/ajustar.py | Ajuste fino con los ejemplos de ONCE (opción de reservar algunos para medir) | Probado con un clasificador de sustitución |
| src/entrenar.py, src/reconocer.py | CNN de 13 clases, entrenamiento y clasificación | Probado: modelos/entrenamiento.json y modelos/metricas.json |
| src/cli.py | `python -m src.cli describir <imagen>` | Probado con `--fen` (sin modelo) |
| src/web.py + src/static/index.html | La web: subir imágenes, PDF o Word; resultados con texto, braille y FEN; lector de voz | Probado en navegador (escritorio, móvil, modo oscuro) |
| src/servicio.py, src/pipeline.py | Lógica de la web: extrae los diagramas de cada archivo y los analiza | Probado |

Uso rápido:

    pip install -r requirements.txt
    python -m src.web --prueba          # la web en http://localhost:8000 sin modelo (solo los 10 ejemplos de ONCE)
    python -m src.web                   # la web con modelos/casillas_v2.pt
    python -m src.web --host 0.0.0.0    # para abrirla desde el móvil: http://IP-del-portátil:8000
    python -m pytest -q
    python -m src.cli describir tests/fixtures/img/tablero2.png --fen r1bqkb1r/pp3ppp/3p4/4p3/4PBn1/2N5/PPP2PPP/R2QKB1R --girado

Entrenar (el dataset chess-positions de Kaggle va en `datos/train`; los modelos descartados (`humo.pt`, `casillas_6000.pt` y `casillas_v3.pt`, que acertaba más tableros pero avisaba de menos errores) están en `datos/modelos_previos`):

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python -m src.preparar_datos --carpeta datos --tableros 500 --salida datos/humo.npz      # prueba de humo
    python -m src.entrenar --npz datos/humo.npz --epocas 1 --salida modelos/humo.pt
    python -m src.preparar_datos --carpeta datos --tableros 6000                      # de verdad -> datos/casillas.npz
    python -m src.entrenar --epocas 8
    python -m src.evaluar --modelo modelos/casillas.pt
    python -m src.ajustar --reservar tablero4.png tablero6.png diagrama23.png --salida modelos/casillas_medida.pt
    python -m src.evaluar --modelo modelos/casillas_medida.pt --solo tablero4.png tablero6.png diagrama23.png
    python -m src.ajustar                                                               # modelo final -> modelos/casillas_v2.pt
    python -m src.web
    python -m src.cli describir tests/fixtures/img/diagrama23.png

tests/fixtures/ejemplos_once.csv tiene la posición, orientación, resaltados y flechas de los 10 ejemplos.
Las FEN de diagrama1, diagrama2 y diagrama5 salen de las imágenes: el texto de audio de ONCE tiene posibles erratas en esos tres.
