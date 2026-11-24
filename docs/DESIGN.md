---
name: Code-switching español–quechua
description: Aplicación local para ver dónde cambia el idioma en un audio, con la verdad y dos modelos sobre los mismos carriles
colors:
  papel: "#fbfbf8"
  superficie: "#ffffff"
  tinta: "#1b2129"
  tinta-2: "#4e5863"
  tinta-3: "#6b7580"
  linea: "#dcdfda"
  linea-fuerte: "#b8bec0"
  sin-habla: "#eceeea"
  pizarra: "#2a3b40"
  pizarra-activo: "#375056"
  tiza: "#ecf1ef"
  tiza-tenue: "#afc0be"
  espanol: "#0072b2"
  espanol-suave: "#e4f0f8"
  quechua: "#e69f00"
  quechua-texto: "#8a5d00"
  quechua-suave: "#fcf1db"
  modelo-no-supervisado: "#6f8890"
  modelo-linea-base: "#b4bcbf"
  ok: "#1e6b47"
  ok-fondo: "#e8f3ec"
  error: "#a3301f"
  error-fondo: "#fbedea"
  ok-borde: "#bcdcc8"
  error-borde: "#eec7bf"
  espanol-borde: "#a9cbe3"
  fila-total: "#f4f5f1"
  riel-ok: "#7fd0a4"
  riel-mal: "#ff9d8a"
  registro-barra: "#56707a"
  registro-ok: "#8fd3ad"
  registro-mal: "#ffab9c"
typography:
  display:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(1.625rem, 2.2vw, 2rem)"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  lead:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(1.125rem, 1.6vw, 1.375rem)"
    fontWeight: 400
    lineHeight: 1.45
  body:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.55
    fontFeature: "tnum"
  intro:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  nav:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 500
    lineHeight: 1.2
  control-small:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.2
  label:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.35
  caption:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.45
  record-clock:
    fontFamily: "Atkinson Hyperlegible Mono, ui-monospace, monospace"
    fontSize: "1.75rem"
    fontWeight: 600
    lineHeight: 1.2
  timecode:
    fontFamily: "Atkinson Hyperlegible Mono, ui-monospace, monospace"
    fontSize: "2rem"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.02em"
rounded:
  unico: "6px"
spacing:
  "1": "4px"
  "2": "8px"
  "3": "12px"
  "4": "16px"
  "5": "20px"
  "6": "24px"
  "8": "32px"
  "10": "40px"
  "12": "48px"
  "16": "64px"
components:
  button-primary:
    backgroundColor: "{colors.tinta}"
    textColor: "{colors.superficie}"
    rounded: "{rounded.unico}"
    padding: "8px 18px"
    height: "44px"
  button-secondary:
    backgroundColor: "{colors.superficie}"
    textColor: "{colors.tinta}"
    rounded: "{rounded.unico}"
    padding: "8px 18px"
    height: "44px"
  segment-selected:
    backgroundColor: "{colors.pizarra}"
    textColor: "{colors.superficie}"
    rounded: "{rounded.unico}"
    padding: "8px 16px"
  nav-item-active:
    backgroundColor: "{colors.pizarra-activo}"
    textColor: "{colors.superficie}"
    rounded: "{rounded.unico}"
    padding: "8px 12px 8px 8px"
  stamp-ok:
    backgroundColor: "{colors.ok-fondo}"
    textColor: "{colors.ok}"
    rounded: "{rounded.unico}"
    padding: "4px 12px 4px 8px"
  phrase-chip-selected:
    backgroundColor: "{colors.pizarra}"
    textColor: "{colors.superficie}"
    rounded: "{rounded.unico}"
    padding: "10px 12px 12px"
---

# Design System: Code-switching español–quechua

## Overview

**Creative North Star: "La cartilla bilingüe"**

La interfaz se comporta como una cartilla de educación intercultural bilingüe: cada idioma tiene su carril, cada vista es una lección numerada y la docente entiende en segundos qué pasa. El papel es blanco y la tinta casi negra; la barra superior es una pizarra con texto de tiza. La riqueza de color vive solo en los datos: los carriles de idioma, las barras de métricas y la onda del audio. El resto del armazón se queda callado.

La pantalla es de trabajo (modo Operate): la navegación, los controles y las tablas son los de siempre. El mundo aporta la tipografía hiperlegible, la paleta, la densidad y una firma: el cambio de idioma dibujado como cambio de carril. Rechaza el panel admin genérico de riel oscuro con tarjetas blancas y tablas planas, y el estudio de audio oscuro con acentos de neón.

**Key Characteristics:**
- Fondo claro pensado para un proyector en un aula iluminada.
- Español en el carril de arriba, quechua en el de abajo, siempre con su nombre.
- Un solo radio en todo el sistema.
- Cifras con números tabulares; monoespaciada solo para código, tiempos de reproducción, teclas, identificadores y el registro.

## Colors

Neutros fríos de papel y pizarra, con los dos colores de idioma Okabe-Ito como únicos tonos saturados.

### Primary
- **Azul español** (#0072b2): carril y tramos del español, foco del teclado, paso en curso del pipeline. Texto blanco encima.
- **Ámbar quechua** (#e69f00): carril y tramos del quechua, siempre con texto de tinta encima. Como texto sobre blanco se usa su sombra oscura #8a5d00.

### Neutral
- **Papel** (#fbfbf8): fondo de página.
- **Superficie** (#ffffff): paneles, tablas, tablero de carriles.
- **Tinta** (#1b2129): texto principal, botón principal, cabezal y marcas de cambio.
- **Tinta 2** (#4e5863) y **Tinta 3** (#6b7580): texto secundario y etiquetas de eje.
- **Línea** (#dcdfda) y **Línea fuerte** (#b8bec0): bordes finos y bordes de controles.
- **Sin habla** (#eceeea): carril vacío y fondo de barras.
- **Pizarra** (#2a3b40), **Pizarra activa** (#375056), **Tiza** (#ecf1ef), **Tiza tenue** (#afc0be): barra de navegación, segmento elegido, registro de la corrida.
- **Modelos**: tinta para el profundo, #6f8890 para el no supervisado, #b4bcbf para la línea base, siempre con su nombre en la cabecera.
- **Bordes de estado** (#bcdcc8, #eec7bf, #a9cbe3): borde de los sellos hecho, fallido y en curso. **Fila total** (#f4f5f1): filas de totales en tablas.
- **Sobre pizarra** (#7fd0a4, #ff9d8a, #56707a, #8fd3ad, #ffab9c): punto de estado de la barra (listo o con fallas) y, en el registro, barra de desplazamiento, líneas correctas y líneas con error.

### Named Rules
**La regla del idioma con nombre.** El azul y el ámbar nunca aparecen solos: cada uso lleva la etiqueta ES/QU o el nombre del idioma al lado.

**La regla de los colores reservados.** El ámbar significa solo quechua. El azul significa español en los datos y, en el armazón, únicamente foco del teclado y paso en curso. Los modelos y las particiones usan neutros.

## Typography

**Display y cuerpo:** Atkinson Hyperlegible Next (servida desde `src/fuentes/`)
**Mono:** Atkinson Hyperlegible Mono (misma familia)

**Character:** una sans de legibilidad máxima, sólida a distancia en un proyector, con su pareja monoespaciada para lo que se lee como código o tiempo.

### Hierarchy
- **Display** (700, clamp(1.625rem, 2.2vw, 2rem), 1.15): título de cada lección, con su número en un cuadro de pizarra, en la misma fila que su bajada.
- **Headline** (700, 1.125rem, 1.3): título de cada bloque dentro de una vista; 32px arriba y 12px abajo.
- **Lead** (400, clamp(1.25rem, 2vw, 1.5rem), 1.45): la frase de resultados calculada desde `resultados/metricas.json`.
- **Body** (400, 16px, 1.55): texto corrido, máximo unos 64 a 70 caracteres por línea.
- **Label** (600, 14px): cabeceras de tabla, leyendas, sellos de estado.
- **Timecode** (mono 600, 2rem): reloj de reproducción.

### Named Rules
**La regla de las cifras tabulares.** Todo número va en la sans con `tabular-nums`; la monoespaciada no se usa como disfraz técnico.

## Layout

Barra superior fija de 56px y contenido a lo ancho, con ancho máximo de 1440px y márgenes de clamp(16px, 3vw, 32px). Es una herramienta, no un documento: cada vista abre con una cabecera de una sola fila (título, bajada y acciones o chequeos a la derecha, sobre una línea fina) y debajo reparte sus bloques en rejilla. Desde 1100px, Datos pone gráfico y tabla a la izquierda y las frases de ejemplo a la derecha, Entrenar pone los pasos a la izquierda y el registro fijo a la derecha, y Resultados va de a dos bloques por fila (la comparación de modelos y los gráficos a lo ancho). Desde 1200px, Probar deja la barra de origen arriba, la lista de frases fija en una columna de 420px y los carriles a la derecha. Ritmo en múltiplos de 4. Las filas de carriles usan una rejilla de tres columnas: nombre (200px), guía ES/QU (28px) y línea de tiempo fluida. Por debajo de 880px la barra baja la navegación a una segunda fila y cada carril pone su nombre arriba; por debajo de 480px las pestañas pierden el cuadro de número. Las tablas anchas se desplazan dentro de su caja, nunca la página.

## Elevation & Depth

Plano por defecto. La profundidad sale de pasos de superficie (papel, superficie, pizarra) y de bordes de 1px. La única sombra es la del globo de detalle de un tramo (`0 4px 14px rgba(27, 33, 41, 0.18)`), que flota sobre los carriles.

## Shapes

Un solo radio de 6px para botones, segmentos, tablas, paneles, chips y sellos. Las marcas de cambio de idioma son círculos (anillo de tinta de 2px sobre blanco), la única forma redonda del sistema junto con el punto de estado de la barra. Los tramos de idioma son rectángulos rectos, sin radio, porque son medidas.

## Components

### Buttons
- **Shape:** radio único (6px), alto 44px.
- **Primary:** fondo tinta, texto blanco; hover #313a45.
- **Secondary:** superficie con borde línea fuerte; hover pasa el borde a tinta.
- **Danger:** borde y texto de error; hover con fondo de error suave.
- **Press:** escala 0.97 en 160ms con ease-out.

### Segmented controls
- **Style:** caja de superficie con borde fino y 4px de relleno; el segmento elegido es pizarra con texto blanco.

### Phrase chips
- **Style:** id de la frase en mono, duración en tabular y una barra de 3px cuyo largo es la duración relativa. El chip elegido se vuelve pizarra.

### Status stamps
- **Style:** sello con borde, ícono de 16px y texto 600. Hecho en verde suave, error en rojo suave, en curso en azul con cifras tabulares, en cola neutro. El estado nunca desaparece; cambia de sello.

### Navigation
- **Style:** barra superior de pizarra, fija; marca a la izquierda, las cuatro lecciones en fila con su número en un cuadro de 22px, y el estado de los modelos a la derecha. La activa rellena el cuadro de tiza y el fondo de pizarra activa; texto 700.

### Lane timeline (componente firma)
Dos carriles por fila (español arriba, quechua abajo) sobre fondo sin habla, tramos de 26px de alto, y cada cambio como una línea vertical de tinta con un círculo de intercambio en medio. Encima, la onda del audio en dos capas (gris y tinta) que se recorta según lo reproducido. Un solo cabezal de 2px cruza la onda y todas las filas. Clic o arrastre para buscar, flechas sobre la onda para avanzar medio segundo, Espacio para reproducir y J/K para cambiar de frase. Al pasar sobre un tramo, un globo muestra su intervalo y cuánto coincide con la verdad.

### Metric bar tables
Cada celda lleva la cifra y una barra de 0 a 1 en el color del modelo; el mejor de los modelos entrenados va en negrita y con contorno. Las barras crecen y las cifras cuentan al entrar, salvo con movimiento reducido.

## Do's and Don'ts

### Do:
- **Do** dibujar el idioma como posición de carril además de color: ES arriba, QU abajo.
- **Do** sacar toda cifra y toda frase comparativa de los archivos de `resultados/`.
- **Do** usar 150 a 250ms con `cubic-bezier(0.23, 1, 0.32, 1)` para el feedback, y una sola entrada animada por vista.
- **Do** respetar `prefers-reduced-motion`: sin revelado de carriles, sin barras que crecen y sin fundido entre lecciones.

### Don't:
- **Don't** usar el ámbar para algo que no sea quechua, ni el azul fuera de español, foco y paso en curso.
- **Don't** poner etiquetas pequeñas en mayúsculas sobre los títulos.
- **Don't** usar animaciones en bucle salvo el punto de estado mientras una corrida está viva.
- **Don't** escribir a mano en la interfaz una cifra o una comparación cualitativa que pueda quedar desactualizada.
