import json

import numpy as np
import pytest
import soundfile as sf

import evaluar
from audio import cargar_audio
from config import ES, IGNORAR, QU
from evaluar import (cambios_de_referencia, etiquetas_desde_tramos, evaluar_particion,
                     evaluar_reales, generar_figuras, leer_etiquetas_audacity)

SR = 16000


def oraculo_de(dir_particion):
    ids = [json.loads(l)['id'] for l in (dir_particion / 'meta.jsonl').read_text(
        encoding='utf-8').splitlines() if l.strip()]
    it = iter(ids)
    return lambda y: np.load(dir_particion / f'{next(it)}.npz')['etiquetas']


def test_leer_audacity(tmp_path):
    (tmp_path / 'a.txt').write_text('0.0\t1.5\tEspañol\n1.5\t3.2\tqu\n\n')
    assert leer_etiquetas_audacity(tmp_path / 'a.txt') == [(0.0, 1.5, ES), (1.5, 3.2, QU)]


def test_leer_audacity_invalido(tmp_path):
    (tmp_path / 'b.txt').write_text('0.0\t1.5\taymara\n')
    with pytest.raises(ValueError, match='Línea 1'):
        leer_etiquetas_audacity(tmp_path / 'b.txt')


def test_leer_audacity_columnas(tmp_path):
    (tmp_path / 'c.txt').write_text('0.0\t1.5\n')
    with pytest.raises(ValueError, match='Línea 1'):
        leer_etiquetas_audacity(tmp_path / 'c.txt')


def test_etiquetas_desde_tramos():
    mascara = np.array([True] * 4 + [False] * 2 + [True] * 4)
    e = etiquetas_desde_tramos([(0.0, 0.03, ES), (0.06, 0.1, QU)], 10, mascara)
    assert e.tolist() == [ES, ES, ES, IGNORAR, IGNORAR, IGNORAR, QU, QU, QU, QU]


def test_cambios_de_referencia():
    meta = {'fragmentos': [{'idioma': 'es', 'inicio': 0, 'fin': 2.0},
                           {'idioma': 'es', 'inicio': 2.0, 'fin': 3.0},
                           {'idioma': 'qu', 'inicio': 3.05, 'fin': 5.0}]}
    assert cambios_de_referencia(meta) == pytest.approx([3.025])


def test_evaluar_particion_oraculo(corpus_falso):
    # un predictor que devuelve exactamente las etiquetas guardadas → exactitud 1 y cero falsos cambios
    r = evaluar_particion(corpus_falso / 'validacion', {'oraculo': oraculo_de(corpus_falso / 'validacion')})
    assert r['oraculo']['trama']['exactitud'] == 1.0
    assert r['oraculo']['falsos_cambios_control'] == 0.0
    assert set(r['oraculo']['por_tipo']) == {'mixta', 'monolingue', 'control'}
    assert set(r['oraculo']['por_duracion']) <= {'<1s', '1-2s', '>=2s'}
    assert r['oraculo']['cambios']['f1'] > 0.9


def test_evaluar_reales_vacio(tmp_path):
    assert evaluar_reales(tmp_path / 'no_existe', {}) is None
    (tmp_path / 'vacia').mkdir()
    assert evaluar_reales(tmp_path / 'vacia', {}) is None


def test_evaluar_reales_par(tmp_path, tono):
    sf.write(tmp_path / 'g.wav', tono(3.0), SR)
    (tmp_path / 'g.txt').write_text('0.0\t3.0\tes\n')
    r = evaluar_reales(tmp_path, {'todo_es': lambda y: np.zeros(1 + len(y) // 160, int)})
    assert r['todo_es']['trama']['exactitud'] == 1.0
    assert r['todo_es']['cambios']['f1'] == 1.0


def test_generar_figuras_humo(tmp_path):
    m1 = {'trama': {'exactitud': .8, 'f1_macro': .8, 'f1_es': .8, 'f1_qu': .8,
                    'matriz': [[8, 2], [2, 8]], 'n': 20}}
    metricas = {'prueba': {'profundo': m1, 'no_supervisado': m1, 'linea_base': m1},
                'reales': None, 'no_supervisado_ventanas': {}, 'generado': 'x'}
    entrenamiento = {'epocas': [{'epoca': 1, 'perdida_entrenamiento': .6, 'perdida_validacion': .5,
                                 'f1_validacion': .7, 'segundos': 1.0},
                                {'epoca': 2, 'perdida_entrenamiento': .4, 'perdida_validacion': .45,
                                 'f1_validacion': .8, 'segundos': 1.0}], 'mejor_epoca': 2}
    ns = {'silueta_por_k': {'2': .5, '3': .4, '4': .3}}
    ejemplos = [{'tipo': 'mixta', 'duracion': 4.0,
                 'verdad': [(0.0, 2.0, ES), (2.0, 4.0, QU)],
                 'profundo': [(0.0, 2.1, ES), (2.1, 4.0, QU)],
                 'no_supervisado': [(0.0, 4.0, ES)]}]
    generar_figuras(metricas, entrenamiento, ns, ejemplos, tmp_path / 'fig')
    for n in ('curvas_entrenamiento', 'matriz_profundo', 'matriz_no_supervisado',
              'silueta_k', 'ejemplos_lineas_tiempo'):
        assert (tmp_path / 'fig' / f'{n}.png').stat().st_size > 0
