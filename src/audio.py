"""Utilidades de audio: carga, normalizacion, deteccion de habla y pausas."""
from pathlib import Path

import librosa
import numpy as np

from config import HOP, SR, VENTANA


class ErrorAudio(ValueError):
    """Audio ilegible, vacio o invalido."""


def n_tramas(n_muestras: int) -> int:
    return 1 + n_muestras // HOP


def cargar_audio(ruta: str | Path) -> np.ndarray:
    """Lee cualquier formato y devuelve mono, 16 kHz, float32."""
    nombre = Path(ruta).name
    try:
        y, _ = librosa.load(ruta, sr=SR, mono=True)
    except Exception as e:
        raise ErrorAudio(f'No se pudo leer el audio: {nombre}') from e
    if len(y) == 0:
        raise ErrorAudio(f'No se pudo leer el audio: {nombre}')
    return y.astype(np.float32)


def normalizar_volumen(y: np.ndarray, objetivo_dbfs: float = -20.0) -> np.ndarray:
    """Lleva el RMS al objetivo en dBFS, recortando a [-1, 1]."""
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2)) if len(y) else 0.0
    if rms == 0:
        return y
    ganancia = 10 ** (objetivo_dbfs / 20) / rms
    return np.clip(y * ganancia, -1.0, 1.0).astype(np.float32)


def recortar_silencios(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    return librosa.effects.trim(y, top_db=top_db)[0]


def mascara_habla(y: np.ndarray, umbral_db: float = -35.0) -> np.ndarray:
    """Por trama: True si el RMS supera umbral_db respecto al RMS maximo."""
    rms = librosa.feature.rms(y=y, frame_length=VENTANA, hop_length=HOP, center=True)[0]
    maximo = rms.max()
    if maximo == 0:
        return np.zeros(len(rms), bool)
    return rms > maximo * 10 ** (umbral_db / 20)


def pausas(y: np.ndarray) -> np.ndarray:
    """Centro (en muestras) de cada racha interior de no-habla."""
    m = mascara_habla(y)
    habla = np.flatnonzero(m)
    if len(habla) == 0:
        return np.array([], dtype=int)
    centros = []
    ini = None
    for i in range(habla[0], habla[-1] + 1):
        if not m[i] and ini is None:
            ini = i
        elif m[i] and ini is not None:
            centros.append((ini + i - 1) / 2 * HOP)
            ini = None
    return np.array(centros, dtype=int)
