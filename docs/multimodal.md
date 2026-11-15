# RAG multimodal: primera muestra local

El chat tiene una única entrada con **Qwen3-VL 2B Instruct como generador común**.
Recupera textos, filas de tablas y figuras, fusiona candidatos y selecciona el
contexto conjunto. No es necesario elegir un modo visual para consultar una imagen.

## Qué está implementado

```text
Pregunta → recuperación híbrida de textos y tablas ─┐
         → BM25 sobre metadatos de figuras ────────┤
         → unión y deduplicación → reranking de representaciones textuales
         → selección y presupuesto → textos + PNG original, si queda seleccionado
         → un mismo Qwen3-VL 2B Instruct → respuesta con fuentes
```

- Cuatro regiones: tabla DWI y notas, mapa sectorial, paneles de calcificaciones
  y diagrama comparativo de léxicos.
- Pool común: hasta 20 candidatos textuales y las regiones visuales pertinentes.
  Filtros PI-RADS/BI-RADS y normalización de alias.
- Qwen3-Reranker-0.6B puntúa juntos párrafos y representaciones de figuras: título,
  términos asociados, leyenda y texto PDF. **No puntúa píxeles**.
- Hasta cuatro fuentes finales, máximo una imagen. Una pregunta que depende de
  una figura disponible reserva una plaza para ella. Se deduplican IDs/padres;
  una imagen y su texto asociado no son el mismo objeto.
- Las fuentes entran completas en el presupuesto o se excluyen y registran;
  no se cortan filas ni notas a mitad de texto.
- El panel de fuentes muestra el PNG enviado, página, coordenadas y acceso al PDF.
  La vista de página completa resalta la región mediante sus coordenadas PDF.
  También presenta pares columna/valor cuando una fuente textual tiene tabla JATS.
- Identidades de PDF/XML, hash del PNG, dimensiones, límites y leyenda conservados.
- Salida generada marcada como pendiente de revisión experta; citas comprobadas
  por ID, sin presentar esa comprobación como fidelidad visual.
- Encoder, reranker y generador se liberan por turnos; exclusión mutua con el laboratorio.

En esta versión **la recuperación es textual sobre metadatos**: título, palabras
asociadas, leyenda y texto PDF disponible. No hay embeddings de imagen, búsqueda
multivector ni OCR integrado. El componente multimodal implementado es la evidencia
visual recuperada y su lectura por el generador.

## Preparación y prueba

Con Ollama local y los dos documentos ya preparados:

```powershell
uv run python scripts/prepare_visual_assets.py
uv run python scripts/prepare_vision_generator.py
uv run --locked --extra embeddings radiology-rag chat
```

En http://127.0.0.1:8765, preguntar directamente: «¿De qué color aparece PZ en la
leyenda del mapa sectorial de PI-RADS?». El contexto puede combinar el PNG con
fragmentos textuales del manual. Los ejemplos son
consultas documentales, no solicitudes de clasificación de pacientes.

El generador es el mismo para preguntas textuales y mixtas. Los controles de encoder
y reranker afectan al pool común; sólo búsqueda mantiene la recuperación textual
sin generar. No se admiten imágenes ni rutas arbitrarias
subidas por el cliente; sólo los recortes del registro local.

## Modelo y presupuesto

Se fija **`qwen3-vl:2b-instruct-q4_K_M`** por digest en
`data/catalog/vision_generator.v1.json`. Se usa el tag Instruct explícito: el alias
genérico `2b` apuntaba a Thinking al preparar esta versión. No intercambiar esos
tags en comparaciones. Modelo con visión y cuantización Q4_K_M; véanse los
[tags oficiales de Ollama](https://ollama.com/library/qwen3-vl/tags).

Límites operativos: una imagen, lado máximo 1008 píxeles, contexto de 8192 tokens,
hasta 384 tokens de salida, temperatura cero y `keep_alive=0`. No hay cargas
simultáneas de encoder, reranker y VLM, ni llamadas a Cloud.

El texto se cuenta con el tokenizer propio de Qwen3-VL Instruct, con revisión y
hashes fijados. Antes de llamar al modelo se reservan 2048 tokens para la imagen,
256 para diferencias de plantilla y el máximo de salida. Es un presupuesto
conservador, **no un conteo exacto de tokens visuales**. El runtime devuelve el
conteo real del prompt; el tamaño de archivo no representa consumo de VRAM.

## Procedencia de las regiones

`data/catalog/visual_regions.v1.json` contiene IDs, páginas físicas y coordenadas
en puntos PDF, con origen arriba a la izquierda. Los recortes se renderizan con
PDFium y se reducen manteniendo la proporción. Se han inspeccionado visualmente
los cuatro recortes; esto comprueba su geometría y legibilidad, no valida respuestas
clínicas.

Las figuras BI-RADS vinculan explícitamente la región PDF seleccionada con su ID
JATS. La relación se ha comprobado en las páginas originales; no se deduce una
página del número de figura ni se asignan páginas a todos los elementos XML.
El texto PDF extraído se distingue de la leyenda JATS y de una transcripción OCR.

El resultado queda en `data/processed/multimodal/v1/manifest.json`, con ID de build
y hashes. PNGs, texto documental y respuestas permanecen fuera de Git. La API sólo
sirve imágenes registradas bajo `/visual/<id>.png`, comprobando ruta y hash.

## Controles y trazas

`POST /api/chat` ejecuta el pipeline común. En `output/unified/runs/<id>.json`
quedan ambos grupos de candidatos, los proxies que puntuó el reranker, el contexto
final, exclusiones por presupuesto, IDs de las imágenes, prompt sin base64 y salida
bruta. La interfaz mantiene el ID de la última imagen para seguimientos breves;
un cambio explícito de sistema o tema evita reutilizar esa imagen automáticamente.

La generación usa [salida estructurada de Ollama](https://docs.ollama.com/capabilities/structured-outputs):
respuesta, `image_source_id` y `text_source_ids` separados por modalidad y restringidos
a los IDs disponibles de su tipo. El modelo selecciona los IDs y la interfaz compone
las citas a partir de ese resultado; son referencias de la respuesta completa, no
una verificación automática por afirmación.
Se valida también en Python. JSON incompleto, IDs ajenos o un ID textual usado como
`image_source_id` se rechazan. Este esquema controla formato y atribución
de tipo, no verifica el apoyo semántico de cada afirmación. La salida JSON original
y el objeto validado se conservan en la traza.

El plan usa reglas acotadas para necesidades visuales y referencias a figuras,
no otro LLM. No selecciona dos generadores: el ranking conjunto selecciona la
evidencia y el VLM común responde. Qwen3 4B y los campos de alcance verificados
se conservan como baseline textual del laboratorio.

Las rutas siguientes son controles aislados del VLM, no el pipeline común:

`POST /api/visual/chat` admite `visual_input`:

| Condición | Qué recibe el mismo VLM |
|---|---|
| `image` — predeterminada en la interfaz | Pregunta, identidad de fuente y PNG; sin leyenda explicativa ni texto PDF |
| `text` | Identidad, leyenda y texto PDF disponibles; ninguna imagen |
| `combined` | Imagen y texto disponibles |

Esto evita confundir el efecto de la imagen con un cambio de modelo. La respuesta
del modelo, la salida tras comprobar IDs de cita, condición, hash de imagen, tiempos,
presupuesto y mensajes sin base64 quedan en `output/visual/runs/<id>.json`.
No se guardan las credenciales ni se duplican las imágenes dentro de esas trazas.

```powershell
uv run --extra embeddings python scripts/check_visual_rag.py
```

Este control pareado usa el mapa sectorial y guarda
`output/visual/paired_smoke.json`. Revisar la salida bruta, los truncamientos y la
imagen antes de interpretar el resultado. Una salida vacía o truncada no demuestra
que el contexto textual sea insuficiente: puede ser un fallo de generación.

En el control final con Instruct, sin imagen el modelo se abstuvo y con imagen
identificó correctamente el color consultado de la leyenda, con cita. Es una
verificación técnica de ese ejemplo, no una validación radiológica.

En una prueba de flechas del diagrama, el mismo modelo confundió el bloque de
destino y cruzó columnas pese a recibir el recorte correcto. Se conserva la salida
en `output/visual/flow_arrow_smoke.json`. Por ello, no se considera validada la
comprensión de diagramas; la interfaz recuerda comprobar flechas y columnas.
Una siguiente comparación debe probar un VLM más capaz o una transcripción
estructurada revisada, conservando el mismo caso de desarrollo.

Pendientes: referencias visuales revisadas, OCR, tablas con combinaciones de celdas
más difíciles, evaluación de flechas y relaciones y una muestra de documentos mayor.
Los diagramas pueden leerse mal aunque la cita sea válida. Las notas fuera del
recorte no son visibles para el modelo en condición `image`.

## Siguiente evolución: recuperación visual

[ColPali](https://arxiv.org/abs/2407.01449) representa páginas mediante múltiples
vectores visuales y los compara con una consulta mediante interacción tardía.
[ColQwen2.5](https://huggingface.co/vidore/colqwen2.5-v0.2) aplica esa estrategia
sobre una base Qwen2.5-VL. Son **recuperadores**, no sustitutos del generador que
responde a la pregunta.

El siguiente experimento será indexar una muestra de páginas con un recuperador
de este tipo y compararlo contra la selección por metadatos, manteniendo el mismo
VLM y las mismas preguntas. Hay que medir memoria y tamaño del índice multivector;
no asumir que cabe sólo por el número de parámetros. Después se podrá estudiar
recuperación híbrida textual/visual y selección de múltiples evidencias.
