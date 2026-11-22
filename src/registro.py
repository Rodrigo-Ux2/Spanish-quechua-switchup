"""
Genera docs/resultados.md: el registro de datos y resultados de la corrida que usan los informes.

Las cifras salen de resultados/*.json y de datos/sintetico/ (estadísticas del corpus). Las pocas
que ningún script guarda (descartes de la preparación, chequeo de uniones R4, recorte por
saturación, picos de memoria y el equipo) quedan escritas aquí, y el texto las marca como medidas
en la corrida de referencia del 2026-11-22.

Uso:  .venv/bin/python src/registro.py
"""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

from config import ES, IGNORAR, QU, RAIZ, RESULTADOS, SINTETICO

SALIDA = RAIZ / 'docs' / 'resultados.md'
REFERENCIA = 'medido en la corrida de referencia del 2026-11-22'
PARTICIONES = ('entrenamiento', 'validacion', 'prueba')
NOMBRES = {'profundo': 'Profundo (CRNN)', 'no_supervisado': 'No supervisado', 'linea_base': 'Línea base'}
TIPOS = {'mixta': 'mixtas', 'monolingue': 'monolingües', 'control': 'de control'}


def _leer(resultados: Path, nombre: str) -> dict:
    with open(resultados / nombre, encoding='utf-8') as f:
        return json.load(f)


def _f(x: float, d: int = 3) -> str:
    return f'{x:.{d}f}'


def generar(resultados: Path = RESULTADOS, sintetico: Path = SINTETICO) -> str:
    """Texto Markdown del registro; FileNotFoundError si falta algún resultado o partición."""
    resultados, sintetico = Path(resultados), Path(sintetico)
    datos = _leer(resultados, 'datos.json')
    ent = _leer(resultados, 'entrenamiento_profundo.json')
    ns = _leer(resultados, 'no_supervisado.json')
    met = _leer(resultados, 'metricas.json')
    tie = _leer(resultados, 'tiempos.json')
    L = []
    w = L.append
    w('# Resultados de la corrida de referencia')
    w('')
    w(f"Registro de los datos y resultados medidos en la corrida completa del pipeline (evaluación generada el {met['generado'][:10]}). "
      'Lo genera `src/registro.py` a partir de los archivos de `resultados/` y de `datos/sintetico/`; este documento es la fuente para los informes. '
      f'Las cifras que ningún script guarda se indican como «{REFERENCIA}».')
    w('')
    w(f'Equipo ({REFERENCIA}): Intel Core Ultra 7 265U, 14 núcleos, solo CPU (PyTorch CPU). Semilla 2026.')
    w('')
    w('## 1. Datos de origen (Common Voice, CC0)')
    w('')
    w('Quechua de Puno (qxp) y español de México (segmentos masculino y femenino). Particiones disjuntas por hablante.')
    w('')
    w('| Idioma | Partición | Clips | Horas | Hablantes |')
    w('|---|---|---:|---:|---:|')
    for idi, nom in (('es', 'Español'), ('qu', 'Quechua')):
        for p in PARTICIONES:
            d = datos[idi][p]
            w(f"| {nom} | {p} | {d['clips']} | {_f(d['horas'], 2)} | {d['hablantes']} |")
    tot = sum(datos[i][p]['clips'] for i in datos for p in datos[i])
    w('')
    w(f'Total: {tot} clips. Descartados por durar menos de 1 s ({REFERENCIA}): 36 de español y 6 de quechua. '
      'Los tar de español no traen `validated.tsv`; se usó la unión de train, dev y test (ruling R1).')
    w('')
    qu = datos['qu']
    w(f"Observaciones: el quechua tiene {_f(qu['entrenamiento']['horas'], 2)} h de entrenamiento frente a "
      f"{_f(datos['es']['entrenamiento']['horas'], 2)} h del español, y solo {qu['validacion']['hablantes']} hablantes en validación "
      f"y {qu['prueba']['hablantes']} en prueba, por lo que las métricas de prueba dependen de pocas voces.")
    w('')
    w('## 2. Corpus sintético')
    w('')
    w('| Partición | Frases | Mixtas | Monolingües | Control | Horas | Tramas ES | Tramas QU | Tramas ignoradas |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for p in PARTICIONES:
        with open(sintetico / p / 'meta.jsonl', encoding='utf-8') as f:
            metas = [json.loads(linea) for linea in f if linea.strip()]
        c = collections.Counter(m['tipo'] for m in metas)
        h = sum(m['duracion'] for m in metas) / 3600
        cnt = np.zeros(3)
        for m in metas:
            e = np.load(sintetico / p / f"{m['id']}.npz")['etiquetas']
            cnt += [(e == ES).sum(), (e == QU).sum(), (e == IGNORAR).sum()]
        fr = cnt / cnt.sum() * 100
        w(f"| {p} | {len(metas)} | {c['mixta']} | {c['monolingue']} | {c['control']} | {_f(h, 2)} | "
          f"{_f(fr[0], 1)} % | {_f(fr[1], 1)} % | {_f(fr[2], 1)} % |")
    w('')
    w('Las tramas ignoradas (pausas y no-habla) no cuentan en la pérdida ni en las métricas. '
      f'Chequeo de uniones (R4, {REFERENCIA}): razón de energía en la unión, mediana 0.06, p95 0.21, máximo 4.22 (un único atípico). '
      f'Muestras recortadas por saturación ({REFERENCIA}): 0.06 %. La escucha manual de 5 frases mixtas y 5 de control queda pendiente.')
    w('')
    w('## 3. Modelo profundo (CRNN)')
    w('')
    parametros = f"{ent['parametros']:,}".replace(',', ' ')
    w(f"Arquitectura: 3 bloques Conv2d + BiLSTM sobre log-mel de 64 bandas, {parametros} parámetros. "
      f"Parada temprana con paciencia 5; mejor época {ent['mejor_epoca']}. Tiempo total {_f(tie['profundo'] / 3600, 2)} h.")
    w('')
    w('| Época | Pérdida entrenamiento | Pérdida validación | F1 validación | Segundos |')
    w('|---:|---:|---:|---:|---:|')
    for e in ent['epocas']:
        mark = ' (mejor)' if e['epoca'] == ent['mejor_epoca'] else ''
        w(f"| {e['epoca']}{mark} | {_f(e['perdida_entrenamiento'], 4)} | {_f(e['perdida_validacion'], 4)} | "
          f"{_f(e['f1_validacion'], 4)} | {_f(e['segundos'], 1)} |")
    w('')
    seg = [e['segundos'] for e in ent['epocas']]
    w('El plan presupuestaba menos de 3 min por época y menos de 60 min en total; en este equipo cada época tardó entre '
      f'{min(seg) / 60:.0f} y {max(seg) / 60:.0f} min. '
      'Se dejó terminar la corrida sin reducir horas ni canales (ver `docs/adr/0001-datos-sinteticos-y-fuentes.md`).')
    w('')
    w('## 4. Modelo no supervisado')
    w('')
    w(f"Ventanas de 500 ms con MFCC, normalización por frase (CMN), estandarización y PCA. Algoritmo elegido: {ns['mejor']}.")
    w('')
    w('Silueta de k-means para cada k, medida en una submuestra de 10 000 ventanas de entrenamiento:')
    w('')
    w('| k | Silueta |')
    w('|---:|---:|')
    for k, v in ns['silueta_por_k'].items():
        w(f'| {k} | {_f(v, 4)} |')
    w('')
    w('Todas las columnas de esta tabla son de la partición de validación:')
    w('')
    w('| Algoritmo | Exactitud val. | F1 macro val. | ARI con idioma val. | ARI con género val. | ARI con hablante val. |')
    w('|---|---:|---:|---:|---:|---:|')
    for a in ('kmeans', 'gmm'):
        v = ns['validacion'][a]
        w(f"| {a} | {_f(v['exactitud'])} | {_f(v['f1_macro'])} | {_f(ns['ari_idioma'][a], 4)} | "
          f"{_f(ns['ari_genero'][a], 4)} | {_f(ns['ari_hablante'][a], 4)} |")
    nv = met['no_supervisado_ventanas']
    w('')
    w(f"En prueba, por ventana ({nv['n_ventanas']} ventanas): ARI {_f(nv['ari'], 4)}, NMI {_f(nv['nmi'], 4)}, "
      f"pureza {_f(nv['pureza'])}. La silueta con k = 2 ({_f(nv['silueta'], 4)}) es la de la submuestra de entrenamiento. "
      'En esta corrida los grupos no se alinean con el idioma: el resultado es prácticamente azar.')
    w('')
    w('## 5. Evaluación en la partición de prueba')
    w('')
    w('Línea base: predecir siempre la clase mayoritaria del entrenamiento (español). Cambios evaluados con tolerancia de ±200 ms.')
    w('')
    w('| Modelo | Exactitud | F1 macro | F1 ES | F1 QU | Precisión cambios | Exhaustividad cambios | F1 cambios | '
      'Falsos cambios en control | Falsos cambios en monolingües |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    P = met['prueba']
    for m, nom in NOMBRES.items():
        t, c = P[m]['trama'], P[m]['cambios']
        w(f"| {nom} | {_f(t['exactitud'])} | {_f(t['f1_macro'])} | {_f(t['f1_es'])} | {_f(t['f1_qu'])} | "
          f"{_f(c['precision'])} | {_f(c['exhaustividad'])} | {_f(c['f1'])} | "
          f"{_f(P[m]['falsos_cambios_control'] * 100, 1)} % | {_f(P[m]['falsos_cambios_monolingue'] * 100, 1)} % |")
    w('')
    w(f"Tramas evaluadas: {P['profundo']['trama']['n']}.")
    w('')
    w('### F1 macro por tipo de frase')
    w('')
    w('| Modelo | Mixta | Monolingüe | Control | F1 cambios en mixtas |')
    w('|---|---:|---:|---:|---:|')
    for m, nom in NOMBRES.items():
        pt = P[m]['por_tipo']
        w(f"| {nom} | {_f(pt['mixta']['trama']['f1_macro'])} | {_f(pt['monolingue']['trama']['f1_macro'])} | "
          f"{_f(pt['control']['trama']['f1_macro'])} | {_f(pt['mixta']['cambios']['f1'])} |")
    w('')
    w('### Exactitud por duración del fragmento')
    w('')
    w('| Modelo | < 1 s | 1–2 s | ≥ 2 s |')
    w('|---|---:|---:|---:|')
    for m, nom in NOMBRES.items():
        d = P[m]['por_duracion']
        w(f"| {nom} | {_f(d['<1s'])} | {_f(d['1-2s'])} | {_f(d['>=2s'])} |")
    w('')
    w('### Matrices de confusión (filas: verdad ES, QU; columnas: predicción ES, QU)')
    w('')
    for m, nom in NOMBRES.items():
        (a, b), (c_, d_) = P[m]['trama']['matriz']
        w(f'- {nom}: ES→ES {a}, ES→QU {b}; QU→ES {c_}, QU→QU {d_}.')
    w('')
    w('Grabaciones reales: ' + ('evaluadas; sus métricas están en `reales` de `resultados/metricas.json`.' if met['reales']
                                else 'no hubo en esta corrida (`reales` vacío).'))
    w('')
    w('## 6. Tiempos de ejecución')
    w('')
    w('| Paso | Segundos | Minutos |')
    w('|---|---:|---:|')
    for k, v in tie.items():
        w(f'| {k} | {_f(v, 1)} | {_f(v / 60, 1)} |')
    w('')
    w(f'Picos de memoria ({REFERENCIA}): síntesis 3.7 GB, con 2.1 GB en disco; entrenamiento profundo 2.2 GB.')
    w('')
    w('## 7. Figuras')
    w('')
    for p in sorted((resultados / 'figuras').glob('*.png')):
        w(f'- `resultados/figuras/{p.name}`')
    w('')
    w('## 8. Lectura de los resultados')
    w('')
    pr, mz = P['profundo'], P['profundo']['trama']['matriz']
    w(f"- El modelo profundo obtiene {_f(pr['trama']['f1_macro'])} de F1 macro, frente a "
      f"{_f(P['linea_base']['trama']['f1_macro'])} de la línea base y {_f(P['no_supervisado']['trama']['f1_macro'])} del no supervisado.")
    w(f"- Hay una brecha entre validación ({_f(ent['epocas'][ent['mejor_epoca'] - 1]['f1_validacion'])}) y prueba "
      f"({_f(pr['trama']['f1_macro'])}): la prueba tiene otros hablantes (solo {qu['prueba']['hablantes']} de quechua) "
      'y la época se eligió por su F1 de validación (parada temprana), lo que favorece a esa partición.')
    qu_es, es_qu = mz[1][0], mz[0][1]
    w(f"- El error dominante es {'QU→ES' if qu_es > es_qu else 'ES→QU'} ({max(qu_es, es_qu)} tramas frente a "
      f"{min(qu_es, es_qu)} de {'ES→QU' if qu_es > es_qu else 'QU→ES'}): el modelo tiende a llamar "
      f"{'español al quechua' if qu_es > es_qu else 'quechua al español'}.")
    dificil = min(TIPOS, key=lambda t: pr['por_tipo'][t]['trama']['f1_macro'])
    w(f"- Las frases {TIPOS[dificil]} son las más difíciles ({_f(pr['por_tipo'][dificil]['trama']['f1_macro'])} de F1 macro); "
      f"localizar el cambio es más difícil que clasificar el idioma (F1 de cambios {_f(pr['cambios']['f1'])}).")
    control, mono = pr['falsos_cambios_control'] * 100, pr['falsos_cambios_monolingue'] * 100
    w(f"- El {_f(control, 1)} % de las frases de control (un idioma, un hablante distinto en cada tramo) recibe al menos un cambio "
      f"falso, frente al {_f(mono, 1)} % de las monolingües (un idioma, un solo hablante, las mismas uniones): la diferencia, "
      f"{_f(control - mono, 1)} puntos, es la parte atribuible al cambio de hablante.")
    w(f"- El agrupamiento sin etiquetas no encuentra la separación por idioma (ARI {_f(nv['ari'], 4)} en las ventanas de prueba): "
      'las MFCC por ventana no bastan sin supervisión.')
    return '\n'.join(L) + '\n'


def main() -> None:
    t0 = time.time()
    try:
        texto = generar()
    except FileNotFoundError as e:
        print(f'Falta {Path(e.filename).name}: corre los pasos de síntesis, entrenamiento y evaluación '
              'antes del registro.', file=sys.stderr)
        sys.exit(1)
    SALIDA.write_text(texto, encoding='utf-8')
    print(f'Registro generado: {SALIDA}')
    print(f'Duración total: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
