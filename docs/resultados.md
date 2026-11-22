# Resultados de la corrida de referencia

Registro de los datos y resultados medidos en la corrida completa del pipeline (evaluación generada el 2026-11-22). Lo genera `src/registro.py` a partir de los archivos de `resultados/` y de `datos/sintetico/`; este documento es la fuente para los informes. Las cifras que ningún script guarda se indican como «medido en la corrida de referencia del 2026-11-22».

Equipo (medido en la corrida de referencia del 2026-11-22): Intel Core Ultra 7 265U, 14 núcleos, solo CPU (PyTorch CPU). Semilla 2026.

## 1. Datos de origen (Common Voice, CC0)

Quechua de Puno (qxp) y español de México (segmentos masculino y femenino). Particiones disjuntas por hablante.

| Idioma | Partición | Clips | Horas | Hablantes |
|---|---|---:|---:|---:|
| Español | entrenamiento | 8682 | 9.76 | 440 |
| Español | validacion | 1010 | 1.14 | 55 |
| Español | prueba | 1072 | 1.24 | 54 |
| Quechua | entrenamiento | 5983 | 6.16 | 61 |
| Quechua | validacion | 1206 | 1.14 | 8 |
| Quechua | prueba | 1266 | 1.19 | 8 |

Total: 19219 clips. Descartados por durar menos de 1 s (medido en la corrida de referencia del 2026-11-22): 36 de español y 6 de quechua. Los tar de español no traen `validated.tsv`; se usó la unión de train, dev y test (ruling R1).

Observaciones: el quechua tiene 6.16 h de entrenamiento frente a 9.76 h del español, y solo 8 hablantes en validación y 8 en prueba, por lo que las métricas de prueba dependen de pocas voces.

## 2. Corpus sintético

| Partición | Frases | Mixtas | Monolingües | Control | Horas | Tramas ES | Tramas QU | Tramas ignoradas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| entrenamiento | 4050 | 2430 | 810 | 810 | 8.73 | 44.9 % | 41.2 % | 13.9 % |
| validacion | 450 | 270 | 90 | 90 | 0.98 | 46.3 % | 40.3 % | 13.4 % |
| prueba | 450 | 270 | 90 | 90 | 0.97 | 44.0 % | 45.2 % | 10.8 % |

Las tramas ignoradas (pausas y no-habla) no cuentan en la pérdida ni en las métricas. Chequeo de uniones (R4, medido en la corrida de referencia del 2026-11-22): razón de energía en la unión, mediana 0.06, p95 0.21, máximo 4.22 (un único atípico). Muestras recortadas por saturación (medido en la corrida de referencia del 2026-11-22): 0.06 %. La escucha manual de 5 frases mixtas y 5 de control queda pendiente.

## 3. Modelo profundo (CRNN)

Arquitectura: 3 bloques Conv2d + BiLSTM sobre log-mel de 64 bandas, 1 109 250 parámetros. Parada temprana con paciencia 5; mejor época 16. Tiempo total 2.30 h.

| Época | Pérdida entrenamiento | Pérdida validación | F1 validación | Segundos |
|---:|---:|---:|---:|---:|
| 1 | 0.5117 | 0.3668 | 0.8334 | 524.7 |
| 2 | 0.3460 | 0.2917 | 0.8773 | 474.4 |
| 3 | 0.2724 | 0.2334 | 0.9059 | 476.7 |
| 4 | 0.2488 | 0.2561 | 0.8948 | 467.0 |
| 5 | 0.2088 | 0.2067 | 0.9203 | 470.5 |
| 6 | 0.1985 | 0.2118 | 0.9113 | 467.9 |
| 7 | 0.1730 | 0.1983 | 0.9198 | 471.7 |
| 8 | 0.1524 | 0.2238 | 0.9097 | 451.3 |
| 9 | 0.1355 | 0.1823 | 0.9286 | 457.8 |
| 10 | 0.1264 | 0.2432 | 0.9081 | 466.5 |
| 11 | 0.1299 | 0.1954 | 0.9305 | 460.2 |
| 12 | 0.1119 | 0.1875 | 0.9312 | 365.2 |
| 13 | 0.1000 | 0.1802 | 0.9388 | 281.3 |
| 14 | 0.0870 | 0.1752 | 0.9382 | 291.0 |
| 15 | 0.0860 | 0.1723 | 0.9395 | 293.1 |
| 16 (mejor) | 0.0795 | 0.1893 | 0.9441 | 295.5 |
| 17 | 0.0795 | 0.2126 | 0.9323 | 301.6 |
| 18 | 0.0707 | 0.2059 | 0.9341 | 308.4 |
| 19 | 0.0766 | 0.1911 | 0.9340 | 320.5 |
| 20 | 0.0733 | 0.1829 | 0.9414 | 317.1 |
| 21 | 0.0657 | 0.1974 | 0.9388 | 319.1 |

El plan presupuestaba menos de 3 min por época y menos de 60 min en total; en este equipo cada época tardó entre 5 y 9 min. Se dejó terminar la corrida sin reducir horas ni canales (ver `docs/adr/0001-datos-sinteticos-y-fuentes.md`).

## 4. Modelo no supervisado

Ventanas de 500 ms con MFCC, normalización por frase (CMN), estandarización y PCA. Algoritmo elegido: kmeans.

Silueta de k-means para cada k, medida en una submuestra de 10 000 ventanas de entrenamiento:

| k | Silueta |
|---:|---:|
| 2 | 0.0849 |
| 3 | 0.0676 |
| 4 | 0.0351 |
| 5 | 0.0395 |
| 6 | 0.0270 |

Todas las columnas de esta tabla son de la partición de validación:

| Algoritmo | Exactitud val. | F1 macro val. | ARI con idioma val. | ARI con género val. | ARI con hablante val. |
|---|---:|---:|---:|---:|---:|
| kmeans | 0.516 | 0.515 | 0.0005 | -0.0001 | 0.0102 |
| gmm | 0.498 | 0.496 | -0.0001 | -0.0001 | 0.0051 |

En prueba, por ventana (12758 ventanas): ARI 0.0035, NMI 0.0024, pureza 0.530. La silueta con k = 2 (0.0849) es la de la submuestra de entrenamiento. En esta corrida los grupos no se alinean con el idioma: el resultado es prácticamente azar.

## 5. Evaluación en la partición de prueba

Línea base: predecir siempre la clase mayoritaria del entrenamiento (español). Cambios evaluados con tolerancia de ±200 ms.

| Modelo | Exactitud | F1 macro | F1 ES | F1 QU | Precisión cambios | Exhaustividad cambios | F1 cambios | Falsos cambios en control | Falsos cambios en monolingües |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Profundo (CRNN) | 0.865 | 0.865 | 0.872 | 0.857 | 0.571 | 0.605 | 0.587 | 44.4 % | 22.2 % |
| No supervisado | 0.531 | 0.523 | 0.461 | 0.585 | 0.115 | 0.368 | 0.175 | 90.0 % | 86.7 % |
| Línea base | 0.494 | 0.330 | 0.661 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0 % | 0.0 % |

Tramas evaluadas: 312185.

### F1 macro por tipo de frase

| Modelo | Mixta | Monolingüe | Control | F1 cambios en mixtas |
|---|---:|---:|---:|---:|
| Profundo (CRNN) | 0.834 | 0.926 | 0.897 | 0.645 |
| No supervisado | 0.505 | 0.558 | 0.542 | 0.243 |
| Línea base | 0.342 | 0.307 | 0.316 | 0.000 |

### Exactitud por duración del fragmento

| Modelo | < 1 s | 1–2 s | ≥ 2 s |
|---|---:|---:|---:|
| Profundo (CRNN) | 0.806 | 0.935 | 0.832 |
| No supervisado | 0.503 | 0.492 | 0.569 |
| Línea base | 0.373 | 0.539 | 0.497 |

### Matrices de confusión (filas: verdad ES, QU; columnas: predicción ES, QU)

- Profundo (CRNN): ES→ES 143768, ES→QU 10304; QU→ES 31782, QU→QU 126331.
- No supervisado: ES→ES 62683, ES→QU 91389; QU→ES 54917, QU→QU 103196.
- Línea base: ES→ES 154068, ES→QU 4; QU→ES 158113, QU→QU 0.

Grabaciones reales: no hubo en esta corrida (`reales` vacío).

## 6. Tiempos de ejecución

| Paso | Segundos | Minutos |
|---|---:|---:|
| preparar | 66.0 | 1.1 |
| sintetizar | 177.0 | 3.0 |
| profundo | 8283.5 | 138.1 |
| no_supervisado | 82.2 | 1.4 |
| evaluar | 62.6 | 1.0 |

Picos de memoria (medido en la corrida de referencia del 2026-11-22): síntesis 3.7 GB, con 2.1 GB en disco; entrenamiento profundo 2.2 GB.

## 7. Figuras

- `resultados/figuras/app_datos.png`
- `resultados/figuras/app_entrenar.png`
- `resultados/figuras/app_probar.png`
- `resultados/figuras/app_probar_error.png`
- `resultados/figuras/app_probar_mp3.png`
- `resultados/figuras/app_resultados.png`
- `resultados/figuras/curvas_entrenamiento.png`
- `resultados/figuras/ejemplos_lineas_tiempo.png`
- `resultados/figuras/matriz_no_supervisado.png`
- `resultados/figuras/matriz_profundo.png`
- `resultados/figuras/silueta_k.png`

## 8. Lectura de los resultados

- El modelo profundo obtiene 0.865 de F1 macro, frente a 0.330 de la línea base y 0.523 del no supervisado.
- Hay una brecha entre validación (0.944) y prueba (0.865): la prueba tiene otros hablantes (solo 8 de quechua) y la época se eligió por su F1 de validación (parada temprana), lo que favorece a esa partición.
- El error dominante es QU→ES (31782 tramas frente a 10304 de ES→QU): el modelo tiende a llamar español al quechua.
- Las frases mixtas son las más difíciles (0.834 de F1 macro); localizar el cambio es más difícil que clasificar el idioma (F1 de cambios 0.587).
- El 44.4 % de las frases de control (un idioma, un hablante distinto en cada tramo) recibe al menos un cambio falso, frente al 22.2 % de las monolingües (un idioma, un solo hablante, las mismas uniones): la diferencia, 22.2 puntos, es la parte atribuible al cambio de hablante.
- El agrupamiento sin etiquetas no encuentra la separación por idioma (ARI 0.0035 en las ventanas de prueba): las MFCC por ventana no bastan sin supervisión.
