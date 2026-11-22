"""Características acústicas: log-mel para el modelo profundo y MFCC por ventana
para el enfoque no supervisado."""
import librosa
import numpy as np

from config import HOP, N_MELS, SR, VENTANA

N_FFT = 512
N_MFCC = 20
VENTANA_TRAMAS = 50   # 500 ms
SALTO_TRAMAS = 25     # 250 ms


def logmel(y: np.ndarray) -> np.ndarray:
    """Log-mel (n_tramas, 64) normalizado por banda (media 0, desviación 1)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, win_length=VENTANA, hop_length=HOP,
        n_mels=N_MELS, center=True)
    x = np.log(mel + 1e-6).T
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
    return x.astype(np.float32)


def mfcc_ventanas(y: np.ndarray, mascara: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(X (N, 80), centros (N,) en tramas): media y desviación de 20 MFCC + deltas
    por ventana de 50 tramas con salto de 25, solo sobre tramas de habla."""
    vacio = (np.zeros((0, 2 * 2 * N_MFCC), np.float32), np.zeros(0, int))
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, win_length=VENTANA,
        hop_length=HOP, n_mels=N_MELS, center=True)
    t = mfcc.shape[1]
    if len(mascara) != t:
        raise ValueError(f'La máscara tiene {len(mascara)} tramas y el audio {t}.')
    if t < VENTANA_TRAMAS or not mascara.any():
        return vacio
    # resta de la media cepstral de las tramas de habla de toda la frase
    mfcc = mfcc - mfcc[:, mascara].mean(axis=1, keepdims=True)
    f = np.concatenate([mfcc, librosa.feature.delta(mfcc)])
    filas, centros = [], []
    for ini in range(0, t - VENTANA_TRAMAS + 1, SALTO_TRAMAS):
        m = mascara[ini:ini + VENTANA_TRAMAS]
        if m.sum() < VENTANA_TRAMAS / 2:
            continue
        v = f[:, ini:ini + VENTANA_TRAMAS][:, m]
        filas.append(np.concatenate([v.mean(axis=1), v.std(axis=1)]))
        centros.append(ini + VENTANA_TRAMAS // 2)
    if not filas:
        return vacio
    return np.array(filas, np.float32), np.array(centros, int)
