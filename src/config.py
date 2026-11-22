"""Constantes y rutas del proyecto de deteccion de code-switching."""
from pathlib import Path

SR = 16000       # frecuencia de muestreo interna (Hz)
HOP = 160        # salto de 10 ms
VENTANA = 400    # ventana de 25 ms
N_MELS = 64

ES, QU, IGNORAR = 0, 1, -100
IDIOMAS = {'es': ES, 'qu': QU}
NOMBRES = {ES: 'Español', QU: 'Quechua'}

SEMILLA = 2026

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / 'datos'
CRUDO = DATOS / 'crudo'
CLIPS = DATOS / 'clips'
SINTETICO = DATOS / 'sintetico'
REALES = DATOS / 'reales'
MODELOS = RAIZ / 'modelos'
RESULTADOS = RAIZ / 'resultados'
FIGURAS = RESULTADOS / 'figuras'
