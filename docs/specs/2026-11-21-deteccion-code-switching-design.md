# Detección de code-switching español–quechua en el habla — diseño

Fecha: 2026-11-21 · Materia: Ciencia de Datos y Machine Learning (UMSS, II/2026) · Área asignada: Reconocimiento de voz

## 1. Intención y porqué

La práctica del segundo parcial (`segundo parcial/Prac2PARDML22026.pdf`) pide un prototipo de aprendizaje
automático que resuelva un problema útil del área asignada **con dos modelos: uno de aprendizaje profundo y
otro de aprendizaje no supervisado**, más un informe con la estructura I–VII del PDF (Introducción, Objetivos,
Marco teórico, Marco referencial, Ingeniería con pre-procesamiento / entrenamiento / validación, Conclusiones,
Bibliografía). La evaluación pesa la aplicación (20 pts), la defensa (15) y las preguntas (15) más que el
informe (10): el prototipo debe funcionar en vivo y cada decisión debe poder explicarse.

Problema (de `docs/Propuestas machine.pdf`): en Cochabamba se mezclan español y quechua en la misma frase
("ya pues, jaku vamos"). Los modelos de voz asumen un solo idioma y fallan justo en el cambio. El sistema
identifica el idioma tramo por tramo (resolución de unos cientos de milisegundos) y marca dónde ocurre el
cambio.

## 2. Objetivos, no-objetivos, anti-objetivos

Objetivos

- Corpus sintético de habla con code-switching, con etiquetas de idioma por trama exactas.
- Modelo profundo supervisado (CNN + BiLSTM) que etiqueta cada trama como español o quechua y detecta los puntos
  de cambio.
- Modelo no supervisado (agrupamiento) que separa los tramos en dos grupos sin usar etiquetas, evaluado contra
  las etiquetas sintéticas.
- Validación que demuestre que el modelo detecta **idioma** y no **cambio de hablante**.
- Aplicación web local: subir o grabar audio y ver la línea de tiempo coloreada por idioma para ambos modelos.
- Informe final en `.docx` con la estructura del PDF.

No-objetivos

- Transcribir el habla (no es ASR).
- Quechua de Cochabamba real en esta entrega: se usa la variedad Collao más cercana disponible (ver §4).
- Grabaciones propias del grupo en esta entrega; quedan para una defensa posterior, y el diseño debe permitir
  agregarlas sin reestructurar (ver §6.6).
- Fine-tuning de modelos grandes pre-entrenados (no hay GPU).
- Pre-informes: se entrega solo el informe final completo; el grupo lo presenta a la docente.

Anti-objetivos (fracaso aunque "funcione")

- Un modelo que acierta porque distingue hablantes, micrófonos o fuentes de datos, no idiomas.
- Un modelo cuyas decisiones el grupo no pueda explicar en la defensa.
- Métricas sobre hablantes que también aparecen en entrenamiento.

## 3. Restricciones

- Máquina: CPU Intel Arrow Lake-U, 14 núcleos, 15 GB de RAM, sin GPU NVIDIA, ~290 GB libres. Todo entrena en CPU.
- Python 3.12 (scikit-learn no tiene paquetes estables para 3.14, igual que en el primer parcial), entorno con
  `uv`.
- Descarga de Common Voice: desde octubre de 2025 solo vía Mozilla Data Collective, con cuenta y aceptación de
  términos en el navegador. Ese paso lo hace un integrante a mano; el programa espera los archivos en una
  carpeta conocida y falla con un mensaje claro si no están.
- Idioma de código, comentarios, interfaz e informe: español, como el primer parcial. Commits en inglés.

## 4. Datos

Fuentes

- Quechua: Common Voice Scripted Speech, **Puno Quechua (qxp)**, ~30 h validadas, CC0. Es quechua Collao, la
  variedad disponible más próxima al quechua sureño boliviano. No se encontró un conjunto de quechua boliviano
  (quh) en Common Voice. Siminchik se descarta por licencia restringida.
- Español: Common Voice español, misma plataforma, para que ambos idiomas compartan condiciones de grabación
  (micrófonos de celular y laptop de voluntarios). El conjunto completo (27.0) pesa 48 GB; se usan los segmentos
  de la versión 26.0 **Mexican Spanish (Male)** y **Mexican Spanish (Female)**, que además son español
  latinoamericano y permiten equilibrar género. Respaldo si no son viables: OpenSLR de español latinoamericano
  (p. ej. SLR75), aceptando mayor riesgo de sesgo de canal y compensándolo con aumentos de canal (§5).
- Los archivos descargados se colocan, sin descomprimir, en `segundo parcial/datos/crudo/`.
- Solo clips validados. El informe documenta variedad, licencia, horas y número de hablantes de cada fuente.

Particiones

- División **por hablante** (`client_id`) en entrenamiento / validación / prueba, aproximadamente 80/10/10 de
  hablantes, sin hablantes compartidos entre particiones. Se separa antes de sintetizar: una frase sintética
  solo mezcla clips de la misma partición.
- Tope de clips por hablante para que pocos hablantes prolíficos no dominen.
- Equilibrio de horas entre idiomas y, cuando la metadata lo permita, de género.
- Volumen de referencia (ajustable por el implementador según tiempos medidos): ~8–10 h de audio sintético de
  entrenamiento, ~1 h de validación, ~1 h de prueba.

## 5. Pre-procesamiento y síntesis del corpus

Por clip

- Conversión a mono 16 kHz, normalización de volumen, recorte de silencios en los extremos.

Fragmentación

- Los clips de Common Voice son oraciones completas; el code-switching real incluye tramos cortos (una palabra).
  Los clips se cortan en fragmentos de 0,5 a 3 s, cortando solo en puntos de baja energía (pausas entre
  palabras) para no partir palabras. Common Voice no trae alineación por palabra, así que la búsqueda de pausas
  es por energía; el umbral es decisión del implementador.

Composición de frases sintéticas (4 a 12 s)

- Mixtas (~60 %): 2 a 5 fragmentos que alternan idioma.
- Monolingües de un solo hablante (~20 %): para que el modelo no asuma que siempre hay un cambio.
- **Control de cambio de hablante** (~20 %): fragmentos de dos o más hablantes distintos **del mismo idioma**,
  etiquetados sin cambio. Es la defensa principal contra el anti-objetivo de detectar hablantes en vez de
  idiomas.
- Uniones con fundido cruzado corto (20–50 ms) y pausas aleatorias breves, iguales en los tres tipos, para que
  la costura no delate el cambio de idioma.
- Aumento aplicado a la frase completa (no por fragmento), para homogeneizar el canal: ruido de fondo leve,
  ganancia aleatoria y, opcionalmente, reverberación simple.
- Semilla fija: el corpus se regenera idéntico.

Etiquetas

- Por trama de 10 ms: español, quechua o "no-habla" (detectada por energía). Las tramas de no-habla se ignoran en
  la pérdida y en las métricas por trama, y se muestran en gris en la aplicación.
- Junto a cada frase se guarda su metadata: tipo (mixta / monolingüe / control), fragmentos con idioma, hablante,
  inicio y fin.

Características

- Log-mel de 64 bandas, ventana de 25 ms, salto de 10 ms, normalizadas por frase. Se precalculan y se guardan en
  disco para no recalcular en cada época.

## 6. Modelos

### 6.1 Modelo profundo (supervisado): CNN + BiLSTM

- Entrada: espectrograma log-mel de la frase completa.
- Arquitectura: tres bloques convolucionales 2D (convolución, batch norm, ReLU, pooling sobre frecuencia y
  reducción temporal moderada), seguidos de una BiLSTM de dos capas y una capa lineal por paso de tiempo con
  salida de dos clases. Resolución de salida de unos 40 ms. Tamaño objetivo: del orden de cientos de miles de
  parámetros, para entrenar en CPU en menos de una hora.
- Entrenamiento: entropía cruzada enmascarada (ignora no-habla), Adam, SpecAugment (máscaras de tiempo y
  frecuencia), parada temprana por F1 macro por trama en validación, semilla fija. Se guarda el mejor modelo y
  la curva de pérdida y F1 por época.
- Post-proceso: suavizado por mediana y duración mínima de tramo de 200 ms. Un punto de cambio es cada
  transición entre tramos de idioma distinto.
- Implementación con PyTorch (CPU).

### 6.2 Modelo no supervisado: agrupamiento de ventanas

- Ventanas de 500 ms con salto de 250 ms, solo sobre habla.
- Características por ventana: media y desviación de 20 MFCC más sus deltas, con normalización por frase (resta
  de media cepstral) para reducir la influencia del hablante y el canal. Estandarización y PCA.
- Algoritmos: K-means y GMM con k = 2, ajustados **sin etiquetas** sobre el conjunto de entrenamiento. Se reporta
  además un barrido de k con silueta, para mostrar si los datos forman naturalmente dos grupos o más.
- Asignación grupo → idioma: solo para evaluar, con el emparejamiento que maximiza el acierto sobre validación
  (algoritmo húngaro). Se deja explícito en el informe que este paso usa etiquetas y no forma parte del
  aprendizaje.
- Post-proceso igual al del modelo profundo, para que los puntos de cambio de ambos modelos sean comparables.
- Análisis obligatorio: si los grupos se alinean con el género o el hablante en vez del idioma (se mide contra
  la metadata), el informe lo reporta como resultado.
- Implementación con scikit-learn y librosa.

### 6.3 Línea base

- Clase mayoritaria por trama, para que las métricas tengan contra qué compararse.

### 6.4 Validación (ambos modelos, sobre la partición de prueba)

- Por trama: exactitud, F1 macro, F1 por idioma, matriz de confusión.
- Puntos de cambio: precisión, exhaustividad y F1 con tolerancia de ±200 ms.
- **Tasa de falsos cambios en las frases de control** (cambio de hablante sin cambio de idioma). Es la métrica
  que responde "¿detecta idioma o hablante?".
- Desglose por tipo de frase (mixta / monolingüe / control) y por duración del fragmento (los tramos cortos son
  los difíciles).
- Solo para el no supervisado: ARI, NMI, pureza y silueta.
- Tabla comparativa profundo vs no supervisado vs línea base, y gráficos (curvas de entrenamiento, matrices de
  confusión, ejemplos de líneas de tiempo con verdad y predicción).

### 6.5 Inferencia sobre audio arbitrario

- Un audio de entrada pasa por el mismo pre-procesamiento por clip (mono, 16 kHz, volumen) y los mismos
  extractores de características que el entrenamiento. Se obtienen los tramos por idioma y los puntos de cambio
  de ambos modelos. Si no hay etiquetas, solo se muestran las predicciones.

### 6.6 Preparado para grabaciones reales (defensa posterior)

- El evaluador acepta una carpeta de audios reales con etiquetas de tramo en un formato de texto simple (inicio,
  fin, idioma por línea; compatible con las etiquetas que exporta Audacity). Si la carpeta existe, se evalúa
  igual que la prueba sintética y se reporta aparte. En esta entrega la carpeta puede estar vacía.

## 7. Aplicación

- Mismo esquema que el primer parcial (`src/servidor.py` del proyecto raíz): servidor HTTP de la biblioteca
  estándar de Python que solo escucha en 127.0.0.1, una página HTML, scripts del proyecto ejecutados desde la
  interfaz con salida en vivo y lista cerrada de pasos permitidos.
- Vistas:
  - **Datos:** horas, hablantes y clips por idioma y partición; ejemplos sintéticos que se pueden escuchar.
  - **Entrenar:** correr preparación, síntesis, entrenamiento de cada modelo y evaluación, con registro en vivo.
  - **Resultados:** tabla comparativa y gráficos de §6.4.
  - **Probar:** subir un archivo o grabar con el micrófono; elegir también una frase de prueba sintética.
    Reproductor con línea de tiempo coloreada por idioma para cada modelo (y la verdad cuando exista), puntos de
    cambio marcados y cursor que avanza con el audio.
- La interfaz pasa por las reglas de diseño visual del usuario antes de escribirse.
- Entradas inválidas (archivo que no es audio, audio vacío o muy corto, modelo sin entrenar) muestran un mensaje
  claro, nunca un error crudo.

## 8. Estructura del proyecto

- Proyecto autocontenido en `segundo parcial/`, con su propio entorno `.venv` (ignorado por git), `README.md`,
  y un atajo `npm run dev` / `npm run setup` como en el primer parcial.
- Carpetas: `src/` (código), `datos/` (audio crudo de Common Voice y corpus sintético, ignorados por git porque
  pesan gigas y se regeneran), `modelos/` (pesos entrenados), `resultados/` (métricas, gráficos, registros),
  `docs/` (este diseño, ADR e informe).
- Cada etapa es un script independiente que se puede correr solo o desde la aplicación: preparar clips, sintetizar
  corpus, entrenar profundo, entrenar no supervisado, evaluar.

## 9. Informe

- `docs/` del proyecto, en `.docx`, con la estructura I–VII del PDF. Contenido de Ingeniería: herramientas y
  versiones, pre-procesamiento y síntesis (§4–5), entrenamiento y ejecución de ambos modelos (§6.1–6.2),
  validación (§6.4) con capturas del prototipo. Las cifras salen de la corrida real, nunca se escriben a mano.
- Las limitaciones van explícitas: corpus sintético, variedad Collao en lugar de la cochabambina, cambio de
  hablante en cada unión.
- Carátula e integrantes iguales a `docs/Propuestas machine.pdf` (UMSS, Ingeniería en Sistemas, cinco
  integrantes, docente Ing. Patricia Erika Rodriguez Bilbao); formato visual siguiendo el informe final del
  primer parcial.

## 10. Enfoques considerados

- **A — elegido:** CNN + BiLSTM entrenada desde cero sobre log-mel, más K-means/GMM sobre estadísticos de MFCC.
  Entrena en CPU en menos de una hora y cada capa se puede explicar en la defensa.
- **B — descartado:** mismo clasificador y mismo agrupamiento sobre representaciones de un modelo multilingüe
  pre-entrenado congelado (XLS-R / MMS). Más preciso, pero la extracción tarda horas en CPU y debilita la
  defensa ("el modelo pre-entrenado ya conocía el quechua").
- **C — descartado:** fine-tuning de un modelo grande de voz. Inviable sin GPU.
- Datos: se eligió corpus sintético sobre grabaciones propias (lentas de etiquetar y escasas para entrenar); las
  grabaciones propias quedan para una defensa posterior (§6.6).

## 11. Decisiones tomadas por el diseñador (corregibles)

- Puno Quechua como fuente de quechua; Common Voice también para el español.
- Proporciones 60/20/20 de frases mixtas / monolingües / control y duraciones de fragmentos y frases.
- Ignorar no-habla en la pérdida en lugar de una tercera clase.
- Tolerancia de ±200 ms para puntos de cambio y duración mínima de tramo de 200 ms.
- Ventanas de 500 ms para el no supervisado.
- Entorno propio en `segundo parcial/` en lugar de reutilizar el del primer parcial.

## 12. Queda a criterio del implementador

- Hiperparámetros exactos (canales, tamaño de la LSTM, tasa de aprendizaje, épocas), dentro del presupuesto de
  menos de una hora de entrenamiento en CPU.
- Umbral de energía para pausas y no-habla; parámetros del aumento.
- Volumen final del corpus según los tiempos medidos.
- Formato de guardado de características y metadata.
- Librería para generar el `.docx` y los gráficos.
