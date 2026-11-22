"""Síntesis del corpus de code-switching: fragmentos de clips reales de un mismo
hablante, unidos con pausa y fundido cruzado, con etiquetas exactas por trama."""
import argparse
import csv
import json
import sys
import time
from multiprocessing import get_context

import numpy as np
import soundfile as sf

from audio import cargar_audio, mascara_habla, n_tramas, pausas
from caracteristicas import logmel
from config import CLIPS, HOP, IDIOMAS, IGNORAR, SEMILLA, SINTETICO, SR

Banco = dict[str, dict[str, list[tuple[np.ndarray, str]]]]

TIPOS = {'mixta': 0.6, 'monolingue': 0.2, 'control': 0.2}
FRAGMENTO_MIN, FRAGMENTO_MAX = int(0.5 * SR), int(3.0 * SR)
FRASE_MIN, FRASE_MAX = 4.0, 12.0
MAX_INTENTOS = 1000
DURACION_MEDIA_FRASE = 8.0  # s; convierte horas en número de frases


def fragmentar(y: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Trozos de 0,5 a 3 s cortados solo en pausas; sin corte válido, se descarta el trozo."""
    cortes = [int(c) for c in pausas(y)] + [len(y)]
    trozos, ini = [], 0
    while ini < len(y):
        validos = [c for c in cortes if FRAGMENTO_MIN <= c - ini <= FRAGMENTO_MAX]
        if validos:
            fin = validos[rng.integers(len(validos))]
            trozos.append(y[ini:fin])
        else:
            siguientes = [c for c in cortes if c > ini]
            fin = siguientes[0] if siguientes else len(y)
        ini = fin
    return trozos


def _fragmentos_de_clip(args: tuple[str, int]) -> list[np.ndarray]:
    ruta, semilla = args
    return fragmentar(cargar_audio(ruta), np.random.default_rng(semilla))


def construir_banco(filas: list[dict], rng: np.random.Generator) -> Banco:
    """Fragmenta cada clip una sola vez. `filas` son de una sola partición."""
    semillas = rng.integers(0, 2**32, len(filas))
    tareas = [(str(CLIPS / f['ruta']), int(s)) for f, s in zip(filas, semillas)]
    with get_context('spawn').Pool() as pool:
        resultados = pool.map(_fragmentos_de_clip, tareas, chunksize=16)
    banco: Banco = {}
    for fila, fragmentos in zip(filas, resultados):
        if not fragmentos:
            continue
        banco.setdefault(fila['idioma'], {}).setdefault(fila['hablante'], []).extend(
            (f, fila['genero']) for f in fragmentos)
    return banco


def _tramo(banco: Banco, idioma: str, hablante: str, rng) -> list[tuple[str, str, np.ndarray, str]]:
    """Un tramo: 1 o 2 fragmentos distintos del mismo idioma y hablante."""
    lista = banco[idioma][hablante]
    n = min(int(rng.integers(1, 3)), len(lista))
    return [(idioma, hablante, *lista[i]) for i in rng.choice(len(lista), n, replace=False)]


def _plan(tipo: str, banco: Banco, rng) -> list[tuple[str, str, np.ndarray, str]]:
    n_tramos = int(rng.integers(2, 6))
    idiomas = sorted(banco)
    idioma = idiomas[rng.integers(len(idiomas))]
    hablantes = {i: sorted(banco[i]) for i in idiomas}
    plan, hablante = [], None
    for k in range(n_tramos):
        if tipo == 'mixta':
            if k:
                idioma = next(i for i in idiomas if i != idioma)
            hablante = hablantes[idioma][rng.integers(len(hablantes[idioma]))]
        elif tipo == 'control':
            otros = [h for h in hablantes[idioma] if h != hablante]
            hablante = otros[rng.integers(len(otros))]
        elif hablante is None:  # monolingüe: un solo hablante en toda la frase
            hablante = hablantes[idioma][rng.integers(len(hablantes[idioma]))]
        plan += _tramo(banco, idioma, hablante, rng)
    return plan


def _unir(plan, rng):
    """Pega los fragmentos con pausa y fundido cruzado; devuelve audio y posiciones reales."""
    audio = plan[0][2].copy()
    pos = [(0, len(audio))]
    for _, _, frag, _ in plan[1:]:
        pausa = int(rng.integers(0, int(0.150 * SR) + 1))
        fundido = int(rng.integers(int(0.020 * SR), int(0.050 * SR) + 1))
        fundido = min(fundido, len(frag), len(audio) + pausa)
        audio = np.concatenate([audio, np.zeros(pausa, np.float32)])
        rampa = np.linspace(0.0, 1.0, fundido, dtype=np.float32)
        solape = audio[len(audio) - fundido:] * (1 - rampa) + frag[:fundido] * rampa
        ini = len(audio) - fundido
        audio = np.concatenate([audio[:ini], solape, frag[fundido:]])
        pos.append((ini, ini + len(frag)))
    return audio.astype(np.float32), pos


def _aumentar(audio: np.ndarray, rng) -> np.ndarray:
    """Ruido gaussiano a SNR de 20 a 40 dB y ganancia de ±6 dB sobre la frase completa."""
    potencia = np.mean(audio.astype(np.float64) ** 2)
    snr = rng.uniform(20, 40)
    ruido = rng.normal(0.0, np.sqrt(potencia / 10 ** (snr / 10)), len(audio))
    ganancia = 10 ** (rng.uniform(-6, 6) / 20)
    return np.clip((audio + ruido) * ganancia, -1.0, 1.0).astype(np.float32)


def _etiquetas(audio, plan, pos) -> np.ndarray:
    # frontera entre fragmentos consecutivos: punto medio entre fin del anterior e inicio del siguiente
    fronteras = [(a[1] + b[0]) / 2 for a, b in zip(pos, pos[1:])]
    valores = np.array([IDIOMAS[p[0]] for p in plan], np.int16)
    centros = np.arange(n_tramas(len(audio))) * HOP
    et = valores[np.searchsorted(fronteras, centros, side='right')]
    et[~mascara_habla(audio)] = IGNORAR
    return et


def componer(tipo: str, banco: Banco, rng: np.random.Generator) -> dict:
    """Una frase sintética con etiquetas por trama. Muestreo por rechazo de la duración."""
    if tipo not in TIPOS:
        raise ValueError(f'Tipo de frase desconocido: {tipo}')
    for _ in range(MAX_INTENTOS):
        plan = _plan(tipo, banco, rng)
        audio, pos = _unir(plan, rng)
        if FRASE_MIN <= len(audio) / SR <= FRASE_MAX:
            break
    else:
        raise ValueError(f'No se logró una frase de {FRASE_MIN:.0f} a {FRASE_MAX:.0f} s '
                         f'de tipo {tipo}: el banco de fragmentos es insuficiente.')
    audio = _aumentar(audio, rng)
    fragmentos = [{'idioma': p[0], 'hablante': p[1], 'genero': p[3],
                   'inicio': a / SR, 'fin': b / SR} for p, (a, b) in zip(plan, pos)]
    return {'audio': audio, 'etiquetas': _etiquetas(audio, plan, pos),
            'meta': {'tipo': tipo, 'duracion': len(audio) / SR, 'fragmentos': fragmentos}}


def generar_particion(particion: str, n_frases: int, semilla: int) -> None:
    rng = np.random.default_rng(semilla)
    with open(CLIPS / 'indice.csv', newline='', encoding='utf-8') as f:
        filas = [r for r in csv.DictReader(f) if r['particion'] == particion]
    banco = construir_banco(filas, rng)
    faltan = [i for i in IDIOMAS if len(banco.get(i, {})) < 2]
    if faltan:
        raise ValueError(f'La partición {particion} necesita al menos 2 hablantes con '
                         f'fragmentos en cada idioma; faltan: {", ".join(faltan)}.')
    n_mixta = round(TIPOS['mixta'] * n_frases)
    n_mono = round(TIPOS['monolingue'] * n_frases)
    tipos = (['mixta'] * n_mixta + ['monolingue'] * n_mono
             + ['control'] * (n_frases - n_mixta - n_mono))
    rng.shuffle(tipos)
    destino = SINTETICO / particion
    destino.mkdir(parents=True, exist_ok=True)
    with open(destino / 'meta.jsonl', 'w', encoding='utf-8') as meta:
        for i, tipo in enumerate(tipos):
            id_ = f'{particion[:3]}_{i:05d}'
            frase = componer(tipo, banco, rng)
            sf.write(destino / f'{id_}.wav', frase['audio'], SR, subtype='PCM_16')
            np.savez_compressed(destino / f'{id_}.npz', logmel=logmel(frase['audio']),
                                etiquetas=frase['etiquetas'])
            meta.write(json.dumps({'id': id_, **frase['meta']}, ensure_ascii=False) + '\n')


def main() -> None:
    t0 = time.time()
    p = argparse.ArgumentParser(description='Sintetiza el corpus de code-switching.')
    p.add_argument('--horas-entrenamiento', type=float, default=9)
    p.add_argument('--horas-validacion', type=float, default=1)
    p.add_argument('--horas-prueba', type=float, default=1)
    a = p.parse_args()
    if not (CLIPS / 'indice.csv').exists():
        print('Primero corre la preparación de datos (src/preparar.py).', file=sys.stderr)
        sys.exit(1)
    horas = {'entrenamiento': a.horas_entrenamiento, 'validacion': a.horas_validacion,
             'prueba': a.horas_prueba}
    total = 0.0
    for k, (particion, h) in enumerate(horas.items()):
        n = max(1, round(h * 3600 / DURACION_MEDIA_FRASE))
        print(f'{particion}: {n} frases')
        generar_particion(particion, n, SEMILLA + k)
        with open(SINTETICO / particion / 'meta.jsonl', encoding='utf-8') as f:
            total += sum(json.loads(l)['duracion'] for l in f)
    print(f'Audio sintetizado: {total / 3600:.2f} h')
    print(f'Duración total: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
