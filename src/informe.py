"""Genera el informe final (.docx) a partir de docs/informe/contenido.md.

Todas las cifras salen de resultados/*.json: el texto lleva marcadores $nombre que
valores() rellena ya formateados. Si falta alguna cifra, generar() falla.
"""
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from string import Template

import pypandoc

from config import RAIZ, RESULTADOS

CONTENIDO = RAIZ / 'docs' / 'informe' / 'contenido.md'
SALIDA = RAIZ / 'docs' / 'INFORME FINAL - DETECCION DE CODE-SWITCHING.docx'
MODELOS = ('profundo', 'no_supervisado', 'linea_base')
PARTICIONES = ('entrenamiento', 'validacion', 'prueba')
TIPOS = ('mixta', 'monolingue', 'control')
CUBOS = {'menos1': '<1s', 'de1a2': '1-2s', 'mas2': '>=2s'}
PASOS = ('preparar', 'sintetizar', 'profundo', 'no_supervisado', 'evaluar')
PAQUETES = {'torch': 'torch', 'sklearn': 'scikit-learn', 'librosa': 'librosa', 'numpy': 'numpy',
            'scipy': 'scipy', 'soundfile': 'soundfile', 'matplotlib': 'matplotlib',
            'pypandoc': 'pypandoc_binary'}


def _leer(resultados: Path, nombre: str) -> dict:
    with open(resultados / nombre, encoding='utf-8') as f:
        return json.load(f)


def _pct(x: float) -> str:
    return f'{x * 100:.1f} %'


def _miles(n: int) -> str:
    return f'{n:,}'.replace(',', ' ')


def _tabla(encabezado: list[str], filas: list[list[str]]) -> str:
    lineas = ['| ' + ' | '.join(encabezado) + ' |', '|' + '---|' * len(encabezado)]
    lineas += ['| ' + ' | '.join(f) + ' |' for f in filas]
    return '\n'.join(lineas)


def _estado_reales(reales: dict | None) -> tuple[str, str]:
    """(texto, tabla) de la evaluación en grabaciones reales; 'omitidos' no es un modelo."""
    if not reales:
        return ('En esta corrida no hubo grabaciones reales: `datos/reales/` no contiene pares '
                'audio y etiquetas. Todas las cifras de este informe describen el corpus '
                'sintético y la evaluación en habla real queda pendiente.'), ''
    modelos = {m: r for m, r in reales.items() if m != 'omitidos'}
    filas = [[m, _pct(r['trama']['f1_macro']), _pct(r['cambios']['f1'])] for m, r in modelos.items()]
    omitidos = len(reales.get('omitidos', []))
    texto = (f'Se evaluaron grabaciones reales de `datos/reales/` ({omitidos} archivos omitidos '
             'por errores de formato).')
    return texto, _tabla(['Modelo', 'F1 macro por trama', 'F1 de cambios'], filas)


def valores(resultados: Path = RESULTADOS) -> dict[str, str]:
    """Todas las cifras del informe, ya formateadas (porcentajes y horas con 1 decimal)."""
    datos = _leer(resultados, 'datos.json')
    metricas = _leer(resultados, 'metricas.json')
    entren = _leer(resultados, 'entrenamiento_profundo.json')
    ns = _leer(resultados, 'no_supervisado.json')
    tiempos = _leer(resultados, 'tiempos.json')
    prueba = metricas['prueba']
    v = {}

    for idioma in ('es', 'qu'):
        for p in PARTICIONES:
            d = datos[idioma][p]
            v[f'clips_{idioma}_{p}'] = _miles(d['clips'])
            v[f'horas_{idioma}_{p}'] = f'{d["horas"]:.1f}'
            v[f'hablantes_{idioma}_{p}'] = _miles(d['hablantes'])
    v['total_clips'] = _miles(sum(datos[i][p]['clips'] for i in datos for p in PARTICIONES))

    epocas = entren['epocas']
    mejor = next(e for e in epocas if e['epoca'] == entren['mejor_epoca'])
    seg = [e['segundos'] for e in epocas]
    v.update(parametros=_miles(entren['parametros']), n_epocas=str(len(epocas)),
             mejor_epoca=str(entren['mejor_epoca']), f1_val_mejor=_pct(mejor['f1_validacion']),
             minutos_epoca_min=f'{min(seg) / 60:.1f}', minutos_epoca_max=f'{max(seg) / 60:.1f}',
             tabla_epocas=_tabla(
                 ['Época', 'Pérdida de entrenamiento', 'Pérdida de validación', 'F1 de validación',
                  'Segundos'],
                 [[f'{e["epoca"]} (mejor)' if e['epoca'] == entren['mejor_epoca'] else str(e['epoca']),
                   f'{e["perdida_entrenamiento"]:.4f}', f'{e["perdida_validacion"]:.4f}',
                   _pct(e['f1_validacion']), f'{e["segundos"]:.1f}'] for e in epocas]))

    v['tramas_prueba'] = _miles(prueba['profundo']['trama']['n'])
    for m in MODELOS:
        r = prueba[m]
        v[f'exactitud_{m}'] = _pct(r['trama']['exactitud'])
        v[f'f1_{m}'] = _pct(r['trama']['f1_macro'])
        v[f'f1_es_{m}'] = _pct(r['trama']['f1_es'])
        v[f'f1_qu_{m}'] = _pct(r['trama']['f1_qu'])
        v[f'prec_cambios_{m}'] = _pct(r['cambios']['precision'])
        v[f'exh_cambios_{m}'] = _pct(r['cambios']['exhaustividad'])
        v[f'f1_cambios_{m}'] = _pct(r['cambios']['f1'])
        v[f'falsos_control_{m}'] = _pct(r['falsos_cambios_control'])
        v[f'falsos_mono_{m}'] = _pct(r['falsos_cambios_monolingue'])
        for t in TIPOS:
            v[f'f1_{t}_{m}'] = _pct(r['por_tipo'][t]['trama']['f1_macro'])
        v[f'f1_cambios_mixta_{m}'] = _pct(r['por_tipo']['mixta']['cambios']['f1'])
        for nombre, cubo in CUBOS.items():
            v[f'exac_{nombre}_{m}'] = _pct(r['por_duracion'][cubo])
        (a, b), (c, d) = r['trama']['matriz']
        v.update({f'conf_es_es_{m}': _miles(a), f'conf_es_qu_{m}': _miles(b),
                  f'conf_qu_es_{m}': _miles(c), f'conf_qu_qu_{m}': _miles(d)})
    profundo = prueba['profundo']
    # Control y monolingüe tienen las mismas uniones; solo el control cambia de hablante.
    efecto = profundo['falsos_cambios_control'] - profundo['falsos_cambios_monolingue']
    v['efecto_hablante'] = f'{efecto * 100:.1f} puntos'
    (_, es_qu), (qu_es, _) = profundo['trama']['matriz']
    v['error_dominante'] = 'quechua→español' if qu_es > es_qu else 'español→quechua'

    for k, s in ns['silueta_por_k'].items():
        v[f'silueta_k{k}'] = f'{s:.4f}'
    for a in ('kmeans', 'gmm'):
        v[f'ns_val_exactitud_{a}'] = _pct(ns['validacion'][a]['exactitud'])
        v[f'ns_val_f1_{a}'] = _pct(ns['validacion'][a]['f1_macro'])
        for ref in ('idioma', 'genero', 'hablante'):
            v[f'ari_{ref}_{a}'] = f'{ns[f"ari_{ref}"][a]:.4f}'
    vent = metricas['no_supervisado_ventanas']
    v.update(mejor_algoritmo=ns['mejor'], ventanas_ari=f'{vent["ari"]:.4f}',
             ventanas_nmi=f'{vent["nmi"]:.4f}', ventanas_pureza=_pct(vent['pureza']),
             ventanas_silueta=f'{vent["silueta"]:.4f}', n_ventanas=_miles(vent['n_ventanas']))

    v['estado_reales'], v['tabla_reales'] = _estado_reales(metricas['reales'])
    v['fecha_evaluacion'] = metricas['generado'][:10]

    for paso in PASOS:
        v[f'segundos_{paso}'] = f'{tiempos[paso]:.1f}'
        v[f'minutos_{paso}'] = f'{tiempos[paso] / 60:.1f}'
    v['horas_profundo'] = f'{tiempos["profundo"] / 3600:.1f}'

    v['python'] = '.'.join(map(str, sys.version_info[:3]))
    v.update({f'version_{k}': version(p) for k, p in PAQUETES.items()})
    return v


def generar(salida: Path = SALIDA) -> Path:
    """Sustituye las cifras (falla si falta alguna) y convierte el Markdown a .docx.

    La tabla de contenido es un campo TOC escrito en contenido.md, después de la carátula
    (con --toc de pandoc saldría antes de ella); Word la actualiza al abrir el archivo.
    """
    texto = Template(CONTENIDO.read_text(encoding='utf-8')).substitute(valores())
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    pypandoc.convert_text(texto, 'docx', format='md', outputfile=str(salida),
                          extra_args=[f'--resource-path={RAIZ}', '--columns=20'])
    return salida


def main() -> None:
    t0 = time.time()
    if not (RESULTADOS / 'metricas.json').exists():
        print('Primero corre la evaluación (src/evaluar.py).', file=sys.stderr)
        sys.exit(1)
    try:
        salida = generar()
    except FileNotFoundError as e:
        print(f'Falta {Path(e.filename).name}: corre los pasos de entrenamiento y evaluación '
              'antes del informe.', file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f'Falta la cifra {e}: corre todos los pasos antes del informe o revisa '
              'docs/informe/contenido.md.', file=sys.stderr)
        sys.exit(1)
    print(f'Informe generado: {salida}')
    print(f'Duración total: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
