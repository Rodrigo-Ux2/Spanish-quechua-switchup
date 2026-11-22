"""Métricas por trama y por punto de cambio."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

from config import ES, QU, IGNORAR


def metricas_trama(ref, pred):
    """Exactitud, F1 y matriz de confusión sobre las tramas con ref != IGNORAR.

    Una pred == IGNORAR sobre habla cuenta como error (se mapea a la clase contraria).
    """
    ref, pred = np.asarray(ref), np.asarray(pred)
    valido = ref != IGNORAR
    r, p = ref[valido], pred[valido]
    p = np.where(p == IGNORAR, 1 - r, p)
    n = int(len(r))
    if n == 0:
        return {'exactitud': 0.0, 'f1_macro': 0.0, 'f1_es': 0.0, 'f1_qu': 0.0,
                'matriz': [[0, 0], [0, 0]], 'n': 0}
    matriz = [[int(((r == i) & (p == j)).sum()) for j in (ES, QU)] for i in (ES, QU)]
    f1_es, f1_qu = f1_score(r, p, labels=[ES, QU], average=None, zero_division=0)
    return {'exactitud': float((r == p).mean()),
            'f1_macro': float((f1_es + f1_qu) / 2),
            'f1_es': float(f1_es), 'f1_qu': float(f1_qu),
            'matriz': matriz, 'n': n}


def contar_cambios(ref, pred, tolerancia=0.2):
    """(vp, fp, fn) con emparejamiento uno a uno, de menor a mayor distancia."""
    pares = sorted((abs(a - b), i, j) for i, a in enumerate(ref)
                   for j, b in enumerate(pred) if abs(a - b) <= tolerancia + 1e-9)
    usados_r, usados_p, vp = set(), set(), 0
    for _, i, j in pares:
        if i not in usados_r and j not in usados_p:
            usados_r.add(i)
            usados_p.add(j)
            vp += 1
    return vp, len(pred) - vp, len(ref) - vp


def prf(vp, fp, fn):
    """Precisión, exhaustividad y F1; sin referencia ni predicción valen 1.0."""
    precision = 1.0 if vp + fp == 0 else vp / (vp + fp)
    exhaustividad = 1.0 if vp + fn == 0 else vp / (vp + fn)
    suma = precision + exhaustividad
    f1 = 0.0 if suma == 0 else 2 * precision * exhaustividad / suma
    return {'precision': precision, 'exhaustividad': exhaustividad, 'f1': f1}


def tasa_falsos_cambios(predicciones):
    """Fracción de frases sin cambio real (control o monolingües) con al menos un cambio predicho."""
    if not predicciones:
        return 0.0
    return sum(1 for c in predicciones if len(c) > 0) / len(predicciones)


def mapear_grupos(grupos, ref):
    """Asigna cada grupo a un idioma maximizando coincidencias con ref.

    Los grupos sobrantes (más de dos) toman el idioma con el que más coinciden.
    """
    grupos, ref = np.asarray(grupos), np.asarray(ref)
    valido = ref != IGNORAR
    g, r = grupos[valido], ref[valido]
    ids = [int(x) for x in np.unique(g)]
    cuentas = np.array([[int(((g == i) & (r == l)).sum()) for l in (ES, QU)] for i in ids])
    if len(ids) == 0:
        return {}
    filas, cols = linear_sum_assignment(cuentas, maximize=True)
    mapa = {ids[f]: (ES, QU)[c] for f, c in zip(filas, cols)}
    for k, i in enumerate(ids):
        mapa.setdefault(i, (ES, QU)[int(np.argmax(cuentas[k]))])
    return mapa
