"""Agrupamiento no supervisado (K-means / GMM, k = 2) de ventanas de MFCC.

Las etiquetas de idioma solo se usan después de ajustar, para mapear grupo → idioma
y evaluar; nunca influyen en el escalador, la PCA, K-means ni el GMM."""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from audio import cargar_audio, mascara_habla, n_tramas
from caracteristicas import VENTANA_TRAMAS, mfcc_ventanas
from config import ES, IGNORAR, MODELOS, QU, RESULTADOS, SEMILLA, SINTETICO
from metricas import mapear_grupos, metricas_trama
from modelo_profundo import ErrorModelo

ALGORITMOS = ('kmeans', 'gmm')


def _meta(dir_particion: Path) -> dict:
    with open(Path(dir_particion) / 'meta.jsonl', encoding='utf-8') as f:
        return {m['id']: m for m in map(json.loads, filter(str.strip, f))}


def ventanas_de_particion(dir_particion: Path, max_ventanas: int | None,
                          semilla: int) -> dict:
    """Ventanas de MFCC de una partición. Si hay más de max_ventanas, submuestrea solo
    con el rng sembrado (nunca según las etiquetas)."""
    dir_particion = Path(dir_particion)
    wavs = sorted(dir_particion.glob('*.wav'))
    if not wavs:
        raise ErrorModelo(f'No hay frases en {dir_particion}. '
                          'Primero genera el corpus sintético (src/sintetizar.py).')
    meta = _meta(dir_particion)
    cols = {'X': [], 'idioma': [], 'genero': [], 'hablante': [], 'frase': []}
    for wav in wavs:
        y = cargar_audio(wav)
        X, centros = mfcc_ventanas(y, mascara_habla(y))
        if len(X) == 0:
            continue
        etiquetas = np.load(wav.with_suffix('.npz'))['etiquetas']
        fragmentos = meta[wav.stem]['fragmentos']
        for fila, c in zip(X, centros):
            ventana = etiquetas[max(c - VENTANA_TRAMAS // 2, 0):c + VENTANA_TRAMAS // 2]
            cuentas = np.bincount(ventana[ventana != IGNORAR], minlength=2)
            t = c * 0.01
            frag = next((f for f in fragmentos if f['inicio'] <= t < f['fin']), {})
            cols['X'].append(fila)
            cols['idioma'].append(int(cuentas.argmax()) if cuentas.sum() else IGNORAR)
            cols['genero'].append(frag.get('genero', ''))
            cols['hablante'].append(frag.get('hablante', ''))
            cols['frase'].append(wav.stem)
    if not cols['X']:
        raise ErrorModelo(f'No hay ventanas de habla en {dir_particion}.')
    out = {'X': np.array(cols['X'], np.float32), 'idioma': np.array(cols['idioma'], int),
           'genero': np.array(cols['genero'], str), 'hablante': np.array(cols['hablante'], str),
           'frase': np.array(cols['frase'], str)}
    if max_ventanas is not None and len(out['X']) > max_ventanas:
        idx = np.sort(np.random.default_rng(semilla).choice(
            len(out['X']), max_ventanas, replace=False))
        out = {k: v[idx] for k, v in out.items()}
    return out


def grupos_de_ventanas(modelo: dict, X: np.ndarray, algoritmo: str) -> np.ndarray:
    """Grupos crudos (sin mapear) de las ventanas X."""
    if len(X) == 0:
        return np.zeros(0, int)
    Z = modelo['pca'].transform(modelo['escalador'].transform(X))
    return modelo[algoritmo].predict(Z).astype(int)


def predecir_no_supervisado(modelo: dict, y: np.ndarray,
                            algoritmo: str | None = None) -> np.ndarray:
    """Idioma por trama: cada trama de habla toma el de la ventana de centro más cercano."""
    algoritmo = algoritmo or modelo['mejor']
    mascara = mascara_habla(y)
    pred = np.full(n_tramas(len(y)), IGNORAR, int)
    X, centros = mfcc_ventanas(y, mascara)
    # sin ventanas válidas (habla muy corta) todo queda IGNORAR; nunca falla
    if len(X) == 0:
        return pred
    mapa = modelo['mapa'][algoritmo]
    idiomas = np.array([mapa[int(g)] for g in grupos_de_ventanas(modelo, X, algoritmo)])
    habla = np.flatnonzero(mascara)
    cercano = np.abs(habla[:, None] - centros[None, :]).argmin(axis=1)
    pred[habla] = idiomas[cercano]
    return pred


def cargar_no_supervisado(ruta: Path = MODELOS / 'no_supervisado.joblib') -> dict:
    if not Path(ruta).exists():
        raise ErrorModelo('El modelo no supervisado no está entrenado. '
                          'Corre el paso "Entrenar modelo no supervisado".')
    return joblib.load(ruta)


def entrenar_no_supervisado(dir_sintetico: Path = SINTETICO, max_ventanas: int = 200_000,
                            semilla: int = SEMILLA, dir_modelos: Path | None = None,
                            dir_resultados: Path | None = None) -> dict:
    dir_modelos = Path(dir_modelos or MODELOS)
    dir_resultados = Path(dir_resultados or RESULTADOS)
    ent = ventanas_de_particion(Path(dir_sintetico) / 'entrenamiento', max_ventanas, semilla)
    val = ventanas_de_particion(Path(dir_sintetico) / 'validacion', None, semilla)

    # --- ajuste sin etiquetas ---
    escalador = StandardScaler().fit(ent['X'])
    Z = escalador.transform(ent['X'])
    pca = PCA(n_components=0.95, random_state=semilla).fit(Z)
    Z = pca.transform(Z)
    kmeans = KMeans(2, n_init=10, random_state=semilla).fit(Z)
    gmm = GaussianMixture(2, covariance_type='full', random_state=semilla).fit(Z)
    sub = np.random.default_rng(semilla).choice(len(Z), min(10_000, len(Z)), replace=False)
    silueta = {k: float(silhouette_score(
        Z[sub], KMeans(k, n_init=10, random_state=semilla).fit_predict(Z)[sub]))
        for k in range(2, 7)}

    # --- a partir de aquí sí se usan etiquetas: mapeo y evaluación ---
    modelo = {'escalador': escalador, 'pca': pca, 'kmeans': kmeans, 'gmm': gmm, 'mapa': {}}
    valido = val['idioma'] != IGNORAR
    grupos = {a: grupos_de_ventanas(modelo, val['X'], a) for a in ALGORITMOS}
    for a in ALGORITMOS:
        modelo['mapa'][a] = {int(g): int(l) for g, l in
                             mapear_grupos(grupos[a], val['idioma']).items()}
    ref, preds = [], {a: [] for a in ALGORITMOS}
    for wav in sorted((Path(dir_sintetico) / 'validacion').glob('*.wav')):
        y = cargar_audio(wav)
        ref.append(np.load(wav.with_suffix('.npz'))['etiquetas'])
        for a in ALGORITMOS:
            preds[a].append(predecir_no_supervisado(modelo, y, a))
    validacion = {a: metricas_trama(np.concatenate(ref), np.concatenate(preds[a]))
                  for a in ALGORITMOS}
    modelo['mejor'] = max(ALGORITMOS, key=lambda a: validacion[a]['f1_macro'])
    resultado = {
        'silueta_por_k': silueta, 'validacion': validacion,
        'ari_idioma': {a: float(adjusted_rand_score(val['idioma'][valido], grupos[a][valido]))
                       for a in ALGORITMOS},
        'ari_genero': {a: float(adjusted_rand_score(val['genero'], grupos[a]))
                       for a in ALGORITMOS},
        'ari_hablante': {a: float(adjusted_rand_score(val['hablante'], grupos[a]))
                         for a in ALGORITMOS},
        'mejor': modelo['mejor']}
    dir_modelos.mkdir(parents=True, exist_ok=True)
    dir_resultados.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, dir_modelos / 'no_supervisado.joblib')
    with open(dir_resultados / 'no_supervisado.json', 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    return resultado


def main() -> None:
    argparse.ArgumentParser(
        description='Entrena el agrupamiento no supervisado K-means / GMM.').parse_args()
    if not (SINTETICO / 'entrenamiento').exists():
        print('Primero genera el corpus sintético (src/sintetizar.py).', file=sys.stderr)
        sys.exit(1)
    t0 = time.time()
    r = entrenar_no_supervisado()
    print(f'Mejor algoritmo: {r["mejor"]} '
          f'(F1 macro {r["validacion"][r["mejor"]]["f1_macro"]:.3f})', flush=True)
    print(f'Duración total: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
