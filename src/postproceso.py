"""Suavizado de etiquetas por trama y extracción de tramos y puntos de cambio."""
import numpy as np
from scipy.ndimage import median_filter

from config import HOP, SR, IGNORAR


def _corridas(v):
    """Corridas de valores iguales de v como listas [inicio, fin_exclusivo]."""
    if len(v) == 0:
        return []
    cortes = np.flatnonzero(np.diff(v)) + 1
    ini = np.concatenate(([0], cortes))
    fin = np.concatenate((cortes, [len(v)]))
    return [[int(a), int(b)] for a, b in zip(ini, fin)]


def suavizar(etiquetas, mediana=37, min_tramas=20):
    """Mediana sobre las tramas de habla y fusión de tramos cortos.

    Las tramas IGNORAR se conservan. Los tramos (corridas de igual idioma sobre
    las tramas de habla) con menos de `min_tramas` toman el idioma del vecino
    más largo; si solo hay un tramo, se deja como está.
    """
    etiquetas = np.asarray(etiquetas)
    habla = np.flatnonzero(etiquetas != IGNORAR)
    if len(habla) == 0:
        return etiquetas.copy()
    v = median_filter(etiquetas[habla], size=mediana, mode='nearest')
    while True:
        corr = _corridas(v)
        if len(corr) < 2:
            break
        largos = [b - a for a, b in corr]
        i = int(np.argmin(largos))
        if largos[i] >= min_tramas:
            break
        vecinos = [j for j in (i - 1, i + 1) if 0 <= j < len(corr)]
        j = max(vecinos, key=lambda k: largos[k])
        v[corr[i][0]:corr[i][1]] = v[corr[j][0]]
    salida = etiquetas.copy()
    salida[habla] = v
    return salida


def tramos(etiquetas):
    """Lista de (inicio_s, fin_s, idioma); el silencio no corta un tramo."""
    etiquetas = np.asarray(etiquetas)
    habla = np.flatnonzero(etiquetas != IGNORAR)
    if len(habla) == 0:
        return []
    seg = HOP / SR
    return [(float(habla[a]) * seg, float(habla[b - 1] + 1) * seg,
             int(etiquetas[habla[a]]))
            for a, b in _corridas(etiquetas[habla])]


def puntos_cambio(tramos):
    """Punto medio entre el fin de un tramo y el inicio del siguiente de otro idioma."""
    return [(a[1] + b[0]) / 2 for a, b in zip(tramos, tramos[1:]) if a[2] != b[2]]
