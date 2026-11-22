import json

from config import RESULTADOS
from registro import REFERENCIA, generar


def test_registro_con_resultados_versionados(tmp_path, corpus_falso):
    # datos/ no se versiona: el corpus mínimo hace de las tres particiones
    for particion, origen in (('entrenamiento', 'entrenamiento'), ('validacion', 'validacion'),
                              ('prueba', 'validacion')):
        (tmp_path / particion).symlink_to(corpus_falso / origen, target_is_directory=True)
    texto = generar(RESULTADOS, tmp_path)
    profundo = json.loads((RESULTADOS / 'metricas.json').read_text(encoding='utf-8'))['prueba']['profundo']
    assert f"{profundo['trama']['f1_macro']:.3f}" in texto
    assert f"{profundo['falsos_cambios_monolingue'] * 100:.1f} %" in texto
    assert REFERENCIA in texto
