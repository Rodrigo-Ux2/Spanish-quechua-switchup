# 0001. Corpus sintético de code-switching y fuentes de datos

## Estado

Aceptada, 2026-11-22.

## Contexto

La práctica pide un detector de idioma por tramo para habla que mezcla español y quechua dentro de la misma
frase. No hay un corpus público de code-switching español–quechua con etiquetas de idioma por trama, y no hay
grabaciones propias del grupo en esta entrega (ver sección 2 de la especificación en
`docs/specs/2026-11-21-deteccion-code-switching-design.md`). Etiquetar a mano a nivel de trama no es viable, y
una etiqueta aproximada contaminaría la métrica. Además, el anti-objetivo central es un modelo que acierte
porque distingue hablantes o fuentes y no idiomas, y la máquina de entrenamiento es solo CPU.

Las únicas fuentes disponibles con la calidad y la licencia necesarias son monolingües: Common Voice en
quechua de Puno (qxp) y en español de México (segmentos masculino y femenino). Todas se usan bajo licencia CC0.
El quechua de Puno es la variedad Collao más cercana disponible al quechua de Cochabamba, que es la meta real.

## Decisión

Se construye un corpus sintético pegando segmentos monolingües de ambos idiomas, de modo que la etiqueta de
idioma por trama es exacta por construcción. Las particiones se hacen por hablante, para que ningún hablante
de prueba aparezca en entrenamiento.

Volúmenes finales de la corrida de referencia, medidos en `resultados/datos.json` y en los archivos
`meta.jsonl` de `datos/sintetico/`:

- Preparación: 19 219 clips en 66 s. Se descartaron los clips de menos de 1 s: 36 de español y 6 de quechua.
- Español: entrenamiento 9.76 h y 440 hablantes; validación 1.14 h y 55; prueba 1.24 h y 54.
- Quechua: entrenamiento 6.16 h y 61 hablantes; validación 1.14 h y 8; prueba 1.19 h y 8.
- Síntesis: 177 s, pico de memoria de 3.7 GB y 2.1 GB en disco. Frases de entrenamiento: 4050 (2430 mixtas,
  810 monolingües y 810 de control), 8.73 h. Validación: 450 frases, 0.98 h. Prueba: 450 frases, 0.97 h.
- Alrededor de 14 % de las tramas de entrenamiento llevan la etiqueta IGNORAR. En entrenamiento, el español ocupa 44.9 % y el
  quechua 41.2 % de las tramas.

Resolución R1 sobre el diseño de los archivos tar. Los tar de español traen
`es-MX_female/{train,dev,test}.tsv` y no incluyen `validated.tsv`. El cargador lee la columna de locale y, cuando
falta `validated.tsv`, usa la unión de train, dev y test.

Chequeo de uniones R4. La razón de energía en la unión tiene mediana 0.06, percentil 95 de 0.21 y máximo 4.22,
con un único valor atípico. Las muestras recortadas (clipping) son 0.06 %. Se considera aceptable.

## Consecuencias

Resultados de la corrida de referencia, que motivan lo que sigue. En prueba, por trama, el modelo profundo
(CRNN de 1 109 250 parámetros) alcanza F1 macro 0.865, la línea base 0.330 y el modelo no supervisado
(kmeans) 0.523. El F1 de puntos de cambio a ±200 ms del modelo profundo es 0.587. En las frases de control,
sin cambio real, el modelo profundo marca falsos cambios en 44 % de los casos y el no supervisado en 90 %. El
no supervisado queda prácticamente al azar: ARI de 0.0005 con el idioma y de 0.0102 con el hablante. No había
grabaciones reales, por lo que ese campo de la evaluación queda nulo.

Consecuencias de la decisión:

- Positivo: etiquetas exactas, evaluación reproducible y partición por hablante que protege contra la fuga de
  hablantes entre conjuntos.
- Desbalance de horas. El quechua tiene unas 6.2 h de entrenamiento frente a unas 9.8 h de español. En
  evaluación hay solo 8 hablantes de quechua en prueba y 8 en validación, así que las métricas por idioma son
  sensibles a pocos hablantes y no deben leerse con más precisión de la que tienen.
- Límite de validez. El habla sintética tiene uniones artificiales y no equivale a code-switching espontáneo.
  Los resultados describen el corpus sintético, no el habla real de Cochabamba. Las grabaciones reales
  quedan para una defensa posterior.
- Tiempo de entrenamiento. El plan presupuestaba menos de 3 min por época y menos de 60 min en total. El
  entrenamiento profundo tomó 8283.5 s (2 h 18 min), con unos 470 s por época en las épocas 1 a 11 y unos 300 s
  en las épocas 12 a 21, parada temprana en la época 21 y mejor época la 16, con F1 de validación 0.9441. Se
  decidió no recortar horas ni canales y dejar terminar la corrida, porque el modelo solo se guarda al final y
  abortar descartaba el trabajo hecho. El coste es que repetir la corrida de referencia en CPU toma cerca de
  2.3 h. Reducir horas o canales queda como alternativa si hay que repetirla.
- Pendiente. El paso 3 de la tarea 8, la escucha manual de 5 frases mixtas y 5 de control, no se hizo. Queda a
  cargo del autor humano. El chequeo numérico R4 la sustituye de forma provisional y no equivale a escuchar el
  audio.

## Adenda (2026-11-22): procedencia de cifras, particiones y tamaño del modelo

- Cifras fijas de la corrida de referencia. Los clips descartados por cortos (36 de español y 6 de quechua),
  el chequeo de uniones R4 (mediana 0.06, p95 0.21, máximo 4.22), el recorte por saturación (0.06 %), los
  picos de memoria (3.7 GB en la síntesis, con 2.1 GB en disco, y 2.2 GB en el entrenamiento profundo) y el
  equipo se midieron una sola vez en la corrida de referencia del 2026-11-22. Ningún script los guarda ni los
  regenera: `src/registro.py` los escribe tal cual en `docs/resultados.md`, marcados como medidos en esa
  corrida, y el informe los presenta igual. El resto de las cifras sale de `resultados/` y se recalcula.
- Partición de cada cifra del no supervisado. Los ARI citados en las consecuencias (0.0005 con el idioma y
  0.0102 con el hablante) son de la partición de validación (`resultados/no_supervisado.json`). En las
  ventanas de la prueba el ARI con el idioma es 0.0035 (`no_supervisado_ventanas` en
  `resultados/metricas.json`). La silueta con k = 2 (0.0849) se mide en una submuestra de 10 000 ventanas de
  entrenamiento, no en la prueba.
- Falsos cambios en frases monolingües. La evaluación mide ahora también las frases monolingües: tienen las
  mismas uniones que las de control pero un solo hablante. El modelo profundo marca falsos cambios en el
  22.2 % de ellas frente al 44.4 % de las de control; la diferencia es la parte atribuible al cambio de
  hablante.
- Tamaño de la CRNN. La especificación (§6.1) pedía un modelo «del orden de cientos de miles de parámetros».
  La CRNN tiene 1 109 250: los hiperparámetros fijados en el plan (canales 32, 64 y 64 y una BiLSTM de dos
  capas con 128 unidades) dan ese tamaño, y la BiLSTM aporta 1 052 672 de ellos. Se acepta la desviación
  porque el modelo ya está entrenado y, igual que con las horas y los canales, reducirlo obliga a repetir
  2.3 h de entrenamiento en CPU. Una BiLSTM de una capa o de 64 unidades lo dejaría dentro del rango previsto.
