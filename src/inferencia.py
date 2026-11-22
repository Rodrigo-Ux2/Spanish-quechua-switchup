"""Inferencia: valida un audio y lo analiza con los dos modelos (profundo y no supervisado)."""
from pathlib import Path

import numpy as np

from audio import cargar_audio, mascara_habla, normalizar_volumen
from config import IDIOMAS, MODELOS, SR
from modelo_no_supervisado import cargar_no_supervisado, predecir_no_supervisado
from modelo_profundo import CRNN, cargar_profundo, predecir_profundo
from postproceso import puntos_cambio, suavizar, tramos

DURACION_MIN_S = 1.0
DURACION_MAX_S = 300.0

_NOMBRE_IDIOMA = {v: k for k, v in IDIOMAS.items()}
_modelos = None  # caché: (profundo, no_supervisado); no se guarda si la carga falla


class ErrorEntrada(ValueError):
    """Audio que no se puede analizar (muy corto, muy largo o sin habla)."""


def validar_audio(y: np.ndarray) -> None:
    if len(y) < DURACION_MIN_S * SR:
        raise ErrorEntrada('El audio dura menos de 1 segundo.')
    if len(y) > DURACION_MAX_S * SR:
        raise ErrorEntrada('El audio dura más de 5 minutos; recórtalo.')
    if not mascara_habla(y).any():
        raise ErrorEntrada('No se detectó habla en el audio.')


def _resumen(etiquetas: np.ndarray) -> dict:
    tr = tramos(etiquetas)
    return {'tramos': [{'inicio': round(a, 3), 'fin': round(b, 3), 'idioma': _NOMBRE_IDIOMA[i]}
                       for a, b, i in tr],
            'cambios': [float(c) for c in puntos_cambio(tr)]}


def analizar_senal(y: np.ndarray, profundo: CRNN, no_supervisado: dict) -> dict:
    pred_profundo = suavizar(predecir_profundo(profundo, y))
    pred_ns = suavizar(predecir_no_supervisado(no_supervisado, y))
    return {'duracion': len(y) / SR,
            'modelos': {'profundo': _resumen(pred_profundo),
                        'no_supervisado': {**_resumen(pred_ns),
                                           'algoritmo': no_supervisado['mejor']}}}


def _cargar_modelos() -> tuple:
    global _modelos
    if _modelos is None:
        _modelos = (cargar_profundo(MODELOS / 'profundo.pt'),
                    cargar_no_supervisado(MODELOS / 'no_supervisado.joblib'))
    return _modelos


def analizar(ruta: Path) -> dict:
    y = normalizar_volumen(cargar_audio(ruta))
    validar_audio(y)
    profundo, no_supervisado = _cargar_modelos()
    return analizar_senal(y, profundo, no_supervisado)
