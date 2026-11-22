# Product

<!-- impeccable:product-schema 1 -->

> Registro inferido del brief de la Task 9 y de la especificación (`docs/specs/2026-11-21-deteccion-code-switching-design.md`, §7); no hubo ronda de entrevista con el usuario. Cada hecho marcado *(inferido)* sale de esos documentos, no de una respuesta directa.

## Platform

web

## Users

- Estudiante autor del proyecto y su grupo, que entrenan los modelos y preparan el informe del segundo parcial *(inferido)*.
- Docente que evalúa el prototipo en vivo: debe poder subir o grabar audio y ver al instante qué tramos son español y cuáles quechua *(spec §1: "el prototipo debe funcionar en vivo")*.

## Product Purpose

Aplicación local que detecta *code-switching* español–quechua en audio. Desde una sola ventana se ven los datos de Common Voice y el corpus sintético, se corre el pipeline (preparar, sintetizar, entrenar profundo, entrenar no supervisado, evaluar, informe) con registro en vivo, se comparan las métricas y se prueba cualquier audio con una línea de tiempo por modelo. Éxito: la docente entiende en segundos qué hace cada modelo y dónde se equivoca.

## Positioning

Compara lado a lado un modelo profundo supervisado (CNN + BiLSTM), un modelo no supervisado (agrupamiento de ventanas MFCC) y la verdad de la frase sintética, sobre la misma línea de tiempo y con el mismo audio sonando.

## Operating Context

- Corre en `127.0.0.1` sobre la laptop del estudiante, solo CPU; el entrenamiento profundo tarda cerca de 2,3 h *(resultados/tiempos.json)*.
- Se usa en dos escenas: horas de entrenamiento (la pestaña queda abierta y se recarga) y la defensa en clase (proyector, demostración rápida).

## Capabilities and Constraints

- Lista cerrada de pasos; el servidor no ejecuta nada fuera de ella.
- Audio de entrada: cualquier formato que decodifique el navegador o grabación de micrófono; se envía como WAV mono de 16 kHz, máximo 20 MB, entre 1 s y 5 min.
- Errores del servidor llegan como texto en español y se muestran tal cual.
- Idioma de la interfaz: español.

## Brand Commitments

- Colores de idioma fijos (Okabe-Ito, aptos para daltónicos): español `#0072B2`, quechua `#E69F00`, no-habla en gris neutro; siempre acompañados de su etiqueta de texto. Son los mismos de los gráficos de `resultados/figuras/`.

## Evidence on Hand

- `resultados/metricas.json`, `resultados/datos.json`, `resultados/tiempos.json`, `resultados/figuras/*.png` de la corrida de referencia.
- 900 frases de prueba en `datos/sintetico/prueba/`.
- No hay datos reales etiquetados en esta corrida (`reales` es `null`); no se deben inventar.

## Product Principles

1. Las cifras salen de `resultados/`, nunca se escriben a mano.
2. Ningún error crudo: cada fallo dice qué pasó y qué paso correr.
3. El color de idioma nunca va solo: siempre con su nombre.

## Accessibility & Inclusion

Contraste AA, foco visible, navegación por teclado y paleta segura para daltónicos *(reglas globales del usuario)*.
