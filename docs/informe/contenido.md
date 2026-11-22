**UNIVERSIDAD MAYOR DE SAN SIMÓN**

**FACULTAD DE CIENCIAS Y TECNOLOGÍA**

**INGENIERÍA EN SISTEMAS**

&nbsp;

**CIENCIA DE DATOS Y MACHINE LEARNING**

&nbsp;

# Informe final: detección de code-switching español–quechua en el habla {.unnumbered .unlisted}

**GESTIÓN: II / 2026**

&nbsp;

**INTEGRANTES:**

Cordova Nigoevic Anelis Melani

Cosio Marañon Luciana

Monje Lucas Rodrigo Andres

Moscoso Pinto Gabriel Alejandro

Vargas Soliz Lucas Brandon

&nbsp;

**DOCENTE:** Ing. Patricia Erika Rodriguez Bilbao

**FECHA:** 09/11/26

&nbsp;

**COCHABAMBA – BOLIVIA**

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**TABLA DE CONTENIDO**

```{=openxml}
<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \o "1-2" \h \z \u </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>I. Introducción</w:t></w:r></w:p>
<w:p><w:r><w:t>II. Objetivos</w:t></w:r></w:p><w:p><w:r><w:t>III. Marco teórico</w:t></w:r></w:p><w:p><w:r><w:t>IV. Marco referencial</w:t></w:r></w:p><w:p><w:r><w:t>V. Ingeniería</w:t></w:r></w:p><w:p><w:r><w:t>VI. Conclusiones</w:t></w:r></w:p><w:p><w:r><w:t>VII. Bibliografía</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# I. Introducción

En Cochabamba es normal mezclar el español y el quechua dentro de la misma frase («ya pues, jaku
vamos»). Los modelos de voz suponen un solo idioma y fallan justo en esos cambios. Este proyecto
construye un prototipo que, dado un audio, decide **tramo por tramo** si se habla español o quechua
y marca **dónde ocurre el cambio de idioma**.

La práctica pide resolver un problema del área de reconocimiento de voz con dos modelos: uno de
aprendizaje profundo y otro de aprendizaje no supervisado. Se construyeron los dos sobre la misma
tarea para poder compararlos con la misma vara:

- **Modelo profundo (supervisado):** una red convolucional y recurrente (CNN + BiLSTM) que predice
  el idioma de cada trama de 10 ms.
- **Modelo no supervisado:** un agrupamiento (K-means, con GMM como alternativa) de ventanas de
  500 ms descritas con MFCC, sin usar etiquetas durante el entrenamiento.
- **Línea base:** predecir siempre el idioma mayoritario del entrenamiento (español). Sirve para
  saber cuánto de cada resultado es mérito del modelo y cuánto es solo la proporción de clases.

No existe un corpus público de code-switching español–quechua con etiquetas por trama, y el grupo
no contaba con grabaciones propias. Por eso el corpus de entrenamiento y de prueba es **sintético**:
se pegan fragmentos monolingües de Common Voice, de modo que la etiqueta de cada trama es exacta por
construcción. Esta decisión condiciona todo el informe y sus límites se detallan en la sección V.6.

El resultado principal es que el modelo profundo alcanza un F1 macro por trama de $f1_profundo en la
partición de prueba, frente a $f1_linea_base de la línea base y $f1_no_supervisado del modelo no
supervisado. Localizar el punto exacto del cambio es bastante más difícil que clasificar el idioma:
el F1 de puntos de cambio a ±200 ms es $f1_cambios_profundo.

# II. Objetivos

## Objetivo general

Desarrollar un prototipo de aprendizaje automático que identifique el idioma (español o quechua)
por tramos de pocos cientos de milisegundos en habla que mezcla ambos idiomas, y que marque los
puntos donde cambia.

## Objetivos específicos

1. Construir un corpus de code-switching con etiquetas de idioma exactas por trama a partir de
   Common Voice, con particiones separadas por hablante.
2. Entrenar un modelo profundo (CNN + BiLSTM) que clasifique cada trama y compararlo con una línea
   base trivial.
3. Entrenar un modelo no supervisado (agrupamiento de ventanas de MFCC) y medir cuánto se alinea
   con el idioma sin haber visto etiquetas.
4. Validar ambos modelos sobre una partición de prueba con hablantes que no aparecen en el
   entrenamiento, con métricas por trama y por punto de cambio.
5. Entregar una aplicación web que permita entrenar, ver los resultados y analizar audios propios.
6. Dejar explícitos los límites de validez de los resultados.

# III. Marco teórico

## Identificación de idioma y code-switching

La **identificación de idioma** (LID) asigna un idioma a un fragmento de audio. El
**code-switching** es la alternancia de dos idiomas dentro de una misma conversación o frase. Los
sistemas de LID clásicos clasifican la frase completa; el code-switching exige clasificar por
tramos y localizar los cambios, es decir, una etiqueta por trama en lugar de una por frase.

## Características acústicas

- **Espectrograma log-mel.** El audio se divide en tramas (ventana de 25 ms, salto de 10 ms), se
  calcula su espectro y se agrupa en bandas de la escala mel, que sigue la percepción humana del
  tono. El logaritmo comprime el rango dinámico. Es la entrada habitual de las redes que procesan
  voz.
- **MFCC.** Los coeficientes cepstrales en las frecuencias mel resumen la forma del espectro con
  pocos números. Sus derivadas temporales (deltas) capturan cómo cambian en el tiempo.

## Redes convolucionales y recurrentes

Una **CRNN** combina capas convolucionales, que extraen patrones locales del espectrograma, con
capas recurrentes, que modelan la secuencia en el tiempo. Bartz et al. (2017) la aplicaron a la
identificación de idioma. Una **LSTM bidireccional** lee la secuencia en los dos sentidos, de modo
que la decisión de cada trama usa el contexto anterior y el posterior.
**SpecAugment** (Park et al., 2019) es un aumento de datos que enmascara bloques de tiempo y de
frecuencia del espectrograma para que la red no dependa de una zona concreta.

## Agrupamiento no supervisado

- **K-means** (MacQueen, 1967) reparte los puntos en k grupos minimizando la distancia a los
  centros de grupo.
- Un **modelo de mezclas gaussianas** (GMM; Reynolds, 2009) ajusta k distribuciones normales y
  asigna cada punto al grupo más probable.
- Antes de agrupar se estandariza y se reduce la dimensión con PCA.

Como no hay etiquetas, el agrupamiento solo puede juzgarse comparando sus grupos con el idioma real
después de entrenar.

## Métricas

- **Exactitud y F1 macro por trama.** El F1 macro promedia el F1 de cada idioma sin ponderar por
  frecuencia, de modo que un modelo que acierta solo la clase mayoritaria queda penalizado.
- **Puntos de cambio.** Un cambio predicho acierta si cae a ±200 ms de uno real. Se informan
  precisión, exhaustividad y F1. En frases sin cambio la precisión y la exhaustividad siguen una
  convención fija para no dividir por cero.
- **Falsos cambios en control.** Proporción de frases de un solo idioma en las que el modelo marca
  al menos un cambio.
- **Índice de Rand ajustado (ARI), información mutua normalizada (NMI) y pureza.** Comparan los
  grupos con las etiquetas reales; un ARI cercano a 0 equivale a agrupar al azar.
- **Silueta.** Mide qué tan separados están los grupos entre sí sin mirar etiquetas.

# IV. Marco referencial

## El problema

El habla de Cochabamba mezcla español y quechua dentro de la misma frase. Un reconocedor de voz o
un sistema de subtitulado que fija un idioma por audio falla en los tramos del otro idioma. Detectar
en qué tramo se habla cada idioma es el primer paso para elegir el modelo adecuado en cada tramo.

## Datos disponibles

El conjunto de referencia para el quechua es **Mozilla Common Voice, conjunto qxp, versión 27.0
(2026-09-11)** (quechua de Puno), y para el español, los segmentos masculino y femenino de español
de México (es-MX) del mismo proyecto (Ardila et al., 2020). Los datos se descargaron de Mozilla Data
Collective y se usan bajo licencia CC0.

Hay una diferencia de variedad que debe tenerse presente: el quechua de Puno pertenece a la
variedad Collao, la más cercana disponible al quechua de Cochabamba, pero no es la misma. Tampoco el
español de México es el boliviano. Los resultados describen el corpus construido, no el habla real
de Cochabamba.

## Justificación del corpus sintético

No hay corpus público de code-switching español–quechua con etiquetas por trama. Etiquetar a mano
a nivel de trama no era viable, y una etiqueta aproximada contaminaría las métricas. La alternativa
elegida fue pegar fragmentos monolingües, lo que da etiquetas exactas. La decisión y sus
consecuencias están registradas en `docs/adr/0001-datos-sinteticos-y-fuentes.md`.

## El riesgo central: aprender al hablante y no el idioma

En el corpus sintético cada unión entre idiomas es también un cambio de hablante. Un modelo podría
acertar detectando el cambio de voz y no el cambio de idioma. Como defensa se incluyeron frases de
**control**: fragmentos de varios hablantes distintos del mismo idioma, etiquetados sin cambio. Si el
modelo solo detectara hablantes, fallaría ahí. Los resultados de la sección V.4 muestran que lo
hace en parte.

# V. Ingeniería

## V.1 Herramientas y versiones

Equipo: Intel Core Ultra 7 265U, 14 núcleos, solo CPU, sin GPU. Semilla global 2026.

| Herramienta | Versión | Uso |
|------------------|---------|---------|
| Python | $python | Lenguaje de todo el pipeline |
| PyTorch | $version_torch | CNN + BiLSTM (Paszke et al., 2019) |
| scikit-learn | $version_sklearn | K-means, GMM, PCA, métricas (Pedregosa et al., 2011) |
| librosa | $version_librosa | Log-mel y MFCC (McFee et al., 2015) |
| NumPy | $version_numpy | Cálculo numérico |
| SciPy | $version_scipy | Filtro de mediana del post-proceso |
| soundfile | $version_soundfile | Lectura y escritura de audio |
| Matplotlib | $version_matplotlib | Figuras de evaluación |
| pypandoc | $version_pypandoc | Generación de este informe en `.docx` |

La aplicación es un servidor HTTP de Python que solo escucha en `127.0.0.1` y una página web sin
dependencias externas. Cada etapa del pipeline es un script independiente en `src/`.

## V.2 Pre-procesamiento del dataset

### Preparación de los clips (`src/preparar.py`)

Cada clip se convierte a mono, 16 kHz, se normaliza el volumen y se recortan los silencios de los
extremos. Se descartaron los clips de menos de 1 s: 36 de español y 6 de quechua. Los archivos de
español no traen `validated.tsv`; el cargador usa la unión de `train.tsv`, `dev.tsv` y `test.tsv`.

La división en entrenamiento, validación y prueba es **por hablante** (80/10/10): ningún hablante
aparece en dos particiones, así que la prueba mide generalización a voces nuevas.

| Idioma | Partición | Clips | Horas | Hablantes |
|------------------|---------|---------:|---------:|---------:|
| Español | entrenamiento | $clips_es_entrenamiento | $horas_es_entrenamiento | $hablantes_es_entrenamiento |
| Español | validación | $clips_es_validacion | $horas_es_validacion | $hablantes_es_validacion |
| Español | prueba | $clips_es_prueba | $horas_es_prueba | $hablantes_es_prueba |
| Quechua | entrenamiento | $clips_qu_entrenamiento | $horas_qu_entrenamiento | $hablantes_qu_entrenamiento |
| Quechua | validación | $clips_qu_validacion | $horas_qu_validacion | $hablantes_qu_validacion |
| Quechua | prueba | $clips_qu_prueba | $horas_qu_prueba | $hablantes_qu_prueba |

En total son $total_clips clips, preparados en $segundos_preparar s. El quechua tiene menos horas de
entrenamiento que el español y solo $hablantes_qu_validacion hablantes en validación y
$hablantes_qu_prueba en prueba, de modo que las métricas de prueba dependen de pocas voces.

### Síntesis del corpus (`src/sintetizar.py`)

Los clips se cortan en fragmentos de 0,5 a 3 s, siempre en puntos de baja energía para no partir
palabras. Con ellos se arman frases de 4 a 12 s de tres tipos, en las proporciones 60 % / 20 % / 20 %:

- **Mixtas:** de 2 a 5 tramos que alternan idioma.
- **Monolingües:** un solo hablante y un solo idioma, para que el modelo no asuma que siempre hay un
  cambio.
- **Control:** varios hablantes del mismo idioma, etiquetadas sin cambio.

Cada unión lleva un fundido cruzado de 20 a 50 ms y una pausa aleatoria de 0 a 150 ms, con el mismo
tratamiento en los tres tipos para que la costura no delate el cambio de idioma. Sobre la frase
completa se aplica ruido gaussiano a una relación señal–ruido de 20 a 40 dB y una ganancia de ±6 dB.
Las frases solo combinan clips de una misma partición.

| Partición | Frases | Mixtas | Monolingües | Control | Horas |
|------------------|---------:|---------:|---------:|---------:|---------:|
| Entrenamiento | 4050 | 2430 | 810 | 810 | 8.73 |
| Validación | 450 | 270 | 90 | 90 | 0.98 |
| Prueba | 450 | 270 | 90 | 90 | 0.97 |

| Partición | Tramas ES | Tramas QU | Tramas ignoradas |
|------------------|---------:|---------:|---------:|
| Entrenamiento | 44.9 % | 41.2 % | 13.9 % |
| Validación | 46.3 % | 40.3 % | 13.4 % |
| Prueba | 44.0 % | 45.2 % | 10.8 % |

La síntesis tardó $segundos_sintetizar s, con un pico de memoria de 3.7 GB y 2.1 GB en disco. Las
tramas ignoradas (pausas y no-habla, detectadas por energía) no cuentan en la pérdida ni en las
métricas.

Chequeo de las uniones: la razón de energía en la unión tiene mediana 0.06, percentil 95 de 0.21 y
máximo 4.22 (un único valor atípico). Las muestras recortadas por saturación son 0.06 %. Este chequeo
numérico **no sustituye** la escucha manual de 5 frases mixtas y 5 de control, que quedó pendiente.

### Etiquetas y características

La etiqueta de cada trama de 10 ms es español, quechua o ignorar (no-habla). El número de tramas de
un audio es `1 + n_muestras // 160`. Para el modelo profundo se calcula un log-mel de 64 bandas,
normalizado por frase (media 0 y desviación 1 por banda). Para el no supervisado se calculan 20 MFCC
con sus deltas, normalizados por frase restando la media (CMN), y por cada ventana de 500 ms (con
salto de 250 ms) la media y la desviación de cada coeficiente: 80 dimensiones por ventana.

## V.3 Entrenamiento y ejecución de los modelos

### Modelo profundo: CNN + BiLSTM (`src/modelo_profundo.py`)

La red tiene $parametros parámetros. La entrada es el log-mel de la frase. Tres bloques
convolucionales (Conv2d de 3×3, BatchNorm y ReLU, con 32, 64 y 64 canales) reducen la frecuencia y
el tiempo; sobre su salida trabaja una BiLSTM de 2 capas y 128 unidades ocultas por sentido, y una
capa lineal da dos logits por posición. La salida se expande a una predicción por cada trama de
10 ms.

El entrenamiento usa entropía cruzada que ignora las tramas de no-habla, el optimizador Adam con
tasa de aprendizaje 0.001, lotes de 32 frases y SpecAugment (máscaras de tiempo y de frecuencia)
sobre las frases de entrenamiento. Se detiene temprano con paciencia 5 sobre el F1 de validación,
con un máximo de 30 épocas, y se conserva la mejor época.

La corrida paró tras $n_epocas épocas y la mejor fue la $mejor_epoca, con F1 de validación
$f1_val_mejor. El entrenamiento tardó **$horas_profundo h** en CPU ($segundos_profundo s), con
entre $minutos_epoca_min y $minutos_epoca_max min por época, y un pico de memoria de 2.2 GB. El plan
original presupuestaba menos de 60 min; se decidió no recortar horas ni canales y dejar terminar la
corrida, porque el modelo solo se guarda al final. Repetir el entrenamiento en CPU toma cerca de esa
cantidad de horas.

$tabla_epocas

![Curvas de entrenamiento del modelo profundo.](resultados/figuras/curvas_entrenamiento.png){width=90%}

### Modelo no supervisado (`src/modelo_no_supervisado.py`)

Las ventanas de MFCC se estandarizan, se reducen con PCA (conservando el 95 % de la varianza) y se
agrupan con K-means y con GMM de covarianza completa, ambos con k = 2. Las etiquetas de idioma, de
género y de hablante no intervienen en el ajuste: solo se usan después para medir cuánto se alinean
los grupos con cada una. Para elegir k se calculó la silueta de K-means entre 2 y 6 grupos.

| k | Silueta |
|---:|---:|
| 2 | $silueta_k2 |
| 3 | $silueta_k3 |
| 4 | $silueta_k4 |
| 5 | $silueta_k5 |
| 6 | $silueta_k6 |

![Silueta según k.](resultados/figuras/silueta_k.png){width=55%}

El entrenamiento tardó $segundos_no_supervisado s. Con k = 2 se compararon los dos algoritmos sobre
la partición de validación, y se eligió **$mejor_algoritmo**:

| Algoritmo | Exactitud val. | F1 macro val. | ARI con idioma | ARI con género | ARI con hablante |
|------------------|---------:|---------:|---------:|---------:|---------:|
| K-means | $ns_val_exactitud_kmeans | $ns_val_f1_kmeans | $ari_idioma_kmeans | $ari_genero_kmeans | $ari_hablante_kmeans |
| GMM | $ns_val_exactitud_gmm | $ns_val_f1_gmm | $ari_idioma_gmm | $ari_genero_gmm | $ari_hablante_gmm |

Para convertir grupos en idiomas se asigna cada grupo a un idioma maximizando las coincidencias en
validación; esto usa las etiquetas solo para nombrar los grupos, no para formarlos.

### Post-proceso

Las dos salidas por trama pasan por el mismo post-proceso: un filtro de mediana de 37 tramas de
10 ms (equivale a 9 salidas de la red profunda) y la fusión de tramos de menos de 200 ms (20 tramas)
con el vecino más largo. Así se evitan cambios espurios de pocas tramas.

### Tiempos de ejecución

| Paso | Segundos | Minutos |
|------------------|---------:|---------:|
| Preparar clips | $segundos_preparar | $minutos_preparar |
| Sintetizar corpus | $segundos_sintetizar | $minutos_sintetizar |
| Entrenar modelo profundo | $segundos_profundo | $minutos_profundo |
| Entrenar modelo no supervisado | $segundos_no_supervisado | $minutos_no_supervisado |
| Evaluar y generar figuras | $segundos_evaluar | $minutos_evaluar |

## V.4 Validación del modelo

La evaluación usa la partición de prueba: 450 frases con hablantes que no están en el entrenamiento
($tramas_prueba tramas con habla). Los cambios se evalúan con tolerancia de ±200 ms. Los resultados
se generaron el $fecha_evaluacion.

### Resultado global

Por trama:

| Modelo | Exactitud | F1 macro | F1 ES | F1 QU |
|------------------|---------:|---------:|---------:|---------:|
| Profundo (CRNN) | ${exactitud_profundo} | ${f1_profundo} | ${f1_es_profundo} | ${f1_qu_profundo} |
| No supervisado | ${exactitud_no_supervisado} | ${f1_no_supervisado} | ${f1_es_no_supervisado} | ${f1_qu_no_supervisado} |
| Línea base | ${exactitud_linea_base} | ${f1_linea_base} | ${f1_es_linea_base} | ${f1_qu_linea_base} |

Por punto de cambio (tolerancia de ±200 ms):

| Modelo | Precisión | Exhaustividad | F1 de cambios | Falsos cambios en control |
|------------------|---------:|---------:|---------:|---------:|
| Profundo (CRNN) | ${prec_cambios_profundo} | ${exh_cambios_profundo} | ${f1_cambios_profundo} | ${falsos_control_profundo} |
| No supervisado | ${prec_cambios_no_supervisado} | ${exh_cambios_no_supervisado} | ${f1_cambios_no_supervisado} | ${falsos_control_no_supervisado} |
| Línea base | ${prec_cambios_linea_base} | ${exh_cambios_linea_base} | ${f1_cambios_linea_base} | ${falsos_control_linea_base} |

La línea base responde siempre español: su exactitud ronda el 50 % y su F1 de quechua es 0. Que su
precisión de cambios sea alta y su exhaustividad nula es un efecto de la convención: nunca predice un
cambio, así que no se equivoca en las frases sin cambio.

![Matriz de confusión del modelo profundo.](resultados/figuras/matriz_profundo.png){width=45%}

La matriz del modelo profundo (filas: verdad; columnas: predicción) es:

| Verdad / Predicción | Español | Quechua |
|---|---:|---:|
| Español | $conf_es_es_profundo | $conf_es_qu_profundo |
| Quechua | $conf_qu_es_profundo | $conf_qu_qu_profundo |

El error dominante es quechua predicho como español: $conf_qu_es_profundo tramas, frente a
$conf_es_qu_profundo de español predicho como quechua.

### Por tipo de frase

| Modelo | Mixta | Monolingüe | Control | F1 cambios en mixtas |
|------------------|---------:|---------:|---------:|---------:|
| Profundo (CRNN) | $f1_mixta_profundo | $f1_monolingue_profundo | $f1_control_profundo | $f1_cambios_mixta_profundo |
| No supervisado | $f1_mixta_no_supervisado | $f1_monolingue_no_supervisado | $f1_control_no_supervisado | $f1_cambios_mixta_no_supervisado |
| Línea base | $f1_mixta_linea_base | $f1_monolingue_linea_base | $f1_control_linea_base | $f1_cambios_mixta_linea_base |

### Por duración del fragmento

| Modelo | Menos de 1 s | De 1 a 2 s | 2 s o más |
|------------------|---------:|---------:|---------:|
| Profundo (CRNN) | $exac_menos1_profundo | $exac_de1a2_profundo | $exac_mas2_profundo |
| No supervisado | $exac_menos1_no_supervisado | $exac_de1a2_no_supervisado | $exac_mas2_no_supervisado |
| Línea base | $exac_menos1_linea_base | $exac_de1a2_linea_base | $exac_mas2_linea_base |

### Modelo no supervisado en la prueba

Por ventana ($n_ventanas ventanas de habla): ARI $ventanas_ari, NMI $ventanas_nmi, pureza
$ventanas_pureza y silueta $ventanas_silueta. Los grupos **no se alinean con el idioma**: el
resultado es prácticamente azar. Las MFCC por ventana no bastan para separar los idiomas sin
supervisión. Se alinean un poco más con el hablante (ARI $ari_hablante_kmeans) que con el idioma
(ARI $ari_idioma_kmeans), aunque ambos valores son cercanos a cero.

![Matriz de confusión del modelo no supervisado.](resultados/figuras/matriz_no_supervisado.png){width=45%}

![Ejemplos de líneas de tiempo: verdad y predicción de cada modelo en una frase mixta, una monolingüe y una de control.](resultados/figuras/ejemplos_lineas_tiempo.png){width=85%}

### Grabaciones reales

$estado_reales

$tabla_reales

### Lectura de los resultados

- El modelo profundo supera con margen a la línea base ($f1_profundo frente a $f1_linea_base de F1
  macro) y al no supervisado ($f1_no_supervisado).
- Hay una brecha entre validación ($f1_val_mejor) y prueba ($f1_profundo): la prueba tiene otros
  hablantes y solo $hablantes_qu_prueba de quechua.
- El error dominante es quechua→español: el modelo tiende a llamar español al quechua.
- Las frases mixtas son las más difíciles ($f1_mixta_profundo). Localizar el cambio es más difícil
  que clasificar el idioma (F1 de cambios $f1_cambios_profundo).
- El $falsos_control_profundo de las frases de control (un solo idioma, varios hablantes) recibe al
  menos un cambio falso: el modelo reacciona en parte al cambio de hablante, que es justo el riesgo
  descrito en la sección IV.
- El agrupamiento sin etiquetas no encuentra la separación por idioma (ARI cercano a 0).

## V.5 Prototipo: la aplicación web

`npm run dev` abre una aplicación local (`src/servidor.py` y `src/app.html`) con cuatro vistas: Datos,
Entrenar, Resultados y Probar. Los resultados de la corrida de referencia ya vienen incluidos, así
que las vistas funcionan sin reentrenar.

**Datos.** Muestra los clips por idioma y partición y las frases sintéticas de prueba.

![Vista Datos.](resultados/figuras/app_datos.png){width=90%}

**Entrenar.** Corre los pasos del pipeline en orden, con el estado de cada uno, su tiempo y la
salida de la consola en vivo. Los pasos son los mismos scripts de `src/`; la lista es cerrada, de
modo que el servidor no ejecuta nada más.

![Vista Entrenar.](resultados/figuras/app_entrenar.png){width=90%}

**Resultados.** Reúne las tablas y las figuras de la validación.

![Vista Resultados.](resultados/figuras/app_resultados.png){width=90%}

**Probar.** Permite elegir una frase de prueba, subir un audio (WAV, MP3, M4A, OGG o FLAC de entre
1 s y 5 min) o grabar con el micrófono. El navegador decodifica el audio y lo envía como WAV mono de
16 kHz, y cada modelo dibuja su línea de tiempo con los tramos de español, de quechua, los silencios
y los cambios de idioma.

![Vista Probar con una frase de prueba.](resultados/figuras/app_probar.png){width=90%}

![Análisis de un audio MP3 subido.](resultados/figuras/app_probar_mp3.png){width=90%}

Los audios inválidos (vacíos, de menos de 1 s, de más de 5 min, sin habla o ilegibles) devuelven un
mensaje claro en español y nunca un traceback.

![Mensaje de error ante un archivo que no se puede leer.](resultados/figuras/app_probar_error.png){width=90%}

## V.6 Limitaciones

1. **Corpus sintético.** Las frases son fragmentos pegados, con uniones artificiales. No equivalen
   a code-switching espontáneo; los resultados describen el corpus sintético, no el habla real.
2. **Variedad Collao en lugar de la cochabambina.** El quechua disponible es el de Puno (qxp), el
   más cercano al de Cochabamba pero distinto. El español es el de México, no el boliviano.
3. **Cambio de hablante en cada unión.** En las frases mixtas cada cambio de idioma es también un
   cambio de hablante. Las frases de control mitigan el riesgo pero no lo eliminan: el
   $falsos_control_profundo de las frases de control recibe un cambio falso.
4. **Pocos hablantes de quechua.** Hay $hablantes_qu_validacion en validación y $hablantes_qu_prueba
   en prueba; las métricas por idioma no deben leerse con más precisión de la que tienen.
5. **Sin evaluación en grabaciones reales.** No hubo grabaciones propias en esta corrida; el
   código para evaluarlas existe (`datos/reales/`, con etiquetas en formato de Audacity) pero no
   se ejercitó con datos reales.
6. **Escucha manual de las uniones pendiente.** La escucha de 5 frases mixtas y 5 de control no se
   hizo; solo hay un chequeo numérico de las uniones.
7. **Grabación con micrófono sin verificar.** La opción de grabar desde el micrófono de la
   aplicación no se verificó con un micrófono real. Sí se probaron la subida de archivos y las
   frases de prueba.
8. **Entrenamiento lento en CPU.** El modelo profundo tarda unas $horas_profundo h en CPU.

# VI. Conclusiones

1. El detector profundo (CNN + BiLSTM) identifica el idioma por trama con un F1 macro de
   $f1_profundo en hablantes nuevos del corpus sintético. Supera a la línea base ($f1_linea_base) y
   al agrupamiento no supervisado ($f1_no_supervisado).
2. Marcar el punto exacto del cambio es más difícil que clasificar la trama: el F1 de cambios es
   $f1_cambios_profundo en toda la prueba y $f1_cambios_mixta_profundo en las frases mixtas.
3. El modelo no es un detector de idioma puro: marca algún cambio falso en el $falsos_control_profundo
   de las frases de un solo idioma con varios hablantes, y confunde más quechua con español que al
   revés. El agrupamiento sin supervisión confirma que el idioma no emerge de las MFCC por ventana
   (ARI $ari_idioma_kmeans).
4. Los resultados valen para el corpus sintético. Afirmar que el sistema funciona con el code-switching
   real de Cochabamba exige grabaciones reales etiquetadas, que quedan para una etapa posterior.
5. Trabajo futuro: evaluar con grabaciones reales (la aplicación y `src/evaluar.py` ya admiten
   `datos/reales/`), escuchar las uniones del corpus, conseguir más hablantes de quechua y probar
   características más ricas para el modelo no supervisado.

# VII. Bibliografía

- Ardila, R., Branson, M., Davis, K., Kohler, M., Meyer, J., Henretty, M., Morais, R., Saunders, L.,
  Tyers, F. y Weber, G. (2020). Common Voice: A Massively-Multilingual Speech Corpus. *Proceedings
  of the Twelfth Language Resources and Evaluation Conference (LREC 2020)*, 4218–4222. European
  Language Resources Association.
- Mozilla Common Voice, conjunto qxp (quechua de Puno), versión 27.0 (2026-09-11). Obtenido de
  Mozilla Data Collective. Licencia CC0.
- Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D. y Le, Q. V. (2019).
  SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Proceedings of
  Interspeech 2019*, 2613–2617.
- Bartz, C., Herold, T., Yang, H. y Meinel, C. (2017). Language Identification Using Deep
  Convolutional Recurrent Neural Networks. *Neural Information Processing (ICONIP 2017)*, Lecture
  Notes in Computer Science 10639, 880–889. Springer.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
  *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1,
  281–297. University of California Press.
- Reynolds, D. A. (2009). Gaussian Mixture Models. En S. Z. Li y A. Jain (eds.), *Encyclopedia of
  Biometrics*, 659–663. Springer.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine
  Learning Research*, 12, 2825–2830.
- Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
  *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 8024–8035.
- McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E. y Nieto, O. (2015).
  librosa: Audio and Music Signal Analysis in Python. *Proceedings of the 14th Python in Science
  Conference (SciPy 2015)*, 18–25.
