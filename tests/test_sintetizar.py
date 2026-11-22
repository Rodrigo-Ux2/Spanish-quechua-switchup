import json

import numpy as np
import pytest

from audio import n_tramas
from config import ES, IGNORAR, QU, SR
from sintetizar import componer, fragmentar, generar_particion


def test_fragmentar_duraciones(tono, silencio):
    y = np.concatenate([tono(1.2), silencio(0.3), tono(0.8), silencio(0.3), tono(4.0)])
    fragmentos = fragmentar(y, np.random.default_rng(0))
    assert fragmentos
    for f in fragmentos:
        assert 0.5 * SR <= len(f) <= 3.0 * SR


@pytest.mark.parametrize('tipo', ['mixta', 'monolingue', 'control'])
def test_componer_reglas(tipo, banco_falso):
    rng = np.random.default_rng(1)
    for _ in range(20):
        f = componer(tipo, banco_falso, rng)
        m = f['meta']
        assert 4.0 <= m['duracion'] <= 12.0
        assert len(f['etiquetas']) == n_tramas(len(f['audio']))
        assert f['etiquetas'].dtype == np.int16
        idiomas = [fr['idioma'] for fr in m['fragmentos']]
        cambios = sum(a != b for a, b in zip(idiomas, idiomas[1:]))
        if tipo == 'mixta':
            assert 1 <= cambios <= 4
        else:
            assert cambios == 0
        if tipo == 'control':
            assert len({fr['hablante'] for fr in m['fragmentos']}) >= 2
        if tipo == 'monolingue':
            assert len({fr['hablante'] for fr in m['fragmentos']}) == 1
        assert set(np.unique(f['etiquetas'])) <= {ES, QU, IGNORAR}
        for a, b in zip(m['fragmentos'], m['fragmentos'][1:]):
            assert a['inicio'] < b['inicio'] and a['fin'] <= m['duracion'] + 1e-6


def test_componer_determinista(banco_falso):
    a = componer('mixta', banco_falso, np.random.default_rng(7))
    b = componer('mixta', banco_falso, np.random.default_rng(7))
    assert np.array_equal(a['audio'], b['audio'])


def test_generar_particion_proporciones(tmp_path, monkeypatch, indice_falso):
    monkeypatch.setattr('sintetizar.SINTETICO', tmp_path)
    generar_particion('entrenamiento', 100, semilla=3)
    tipos = [json.loads(l)['tipo'] for l in (tmp_path / 'entrenamiento' / 'meta.jsonl').open()]
    assert len(tipos) == 100 and 50 <= tipos.count('mixta') <= 70
    assert len(list((tmp_path / 'entrenamiento').glob('*.npz'))) == 100
    z = np.load(next((tmp_path / 'entrenamiento').glob('*.npz')))
    assert z['logmel'].shape[1] == 64 and len(z['logmel']) == len(z['etiquetas'])
