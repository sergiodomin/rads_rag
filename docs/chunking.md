# Notebook 02: parsing y chunking

## Ejecución en VSCode

Desde la raíz: `uv sync --locked`, después `uv run python scripts/prepare_tokenizer.py`
sólo si falta el tokenizer. Abrir `notebooks/02_parsing_and_chunking.ipynb`, seleccionar
`.venv/Scripts/python.exe` y ejecutar de arriba abajo. No hace falta JupyterLab ni clave
de Ollama. El notebook lee PDF y manifiesto directamente; no depende de ejecutar 01.
El catálogo selecciona PI-RADS v2.1 por defecto; fijar `DOCUMENT_ID="esur-2012"`
para repetir el experimento histórico. `uv run radiology-rag process` publica el
mismo baseline que 02 y verifica sus artefactos al repetir una huella.

El tokenizer es `intfloat/multilingual-e5-base`, commit
`d128750597153bb5987e10b1c3493a34e5a4502a`, fichero `tokenizer.json`, SHA-256
`62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626`.
Son reglas/vocabulario y postprocesado, **no pesos del encoder**. La librería Tokenizers
se fija en `uv.lock`, sin adoptar la versión 1 prerelease con cambios de API.

## Qué implementa

- Diez microreferencias transcritas de imágenes, con coordenadas y hash, para diagnosticar
  extracción por defecto frente a tolerancia de espacios 1. Incluye un control de
  contenido de imagen ausente. No es un gold standard validado ni un conjunto de QA.
- Mapas ligados al hash: PI-RADS v2.1 (76 páginas, una columna, cuadros de evaluación
  y léxico por página) y el PDF histórico ESUR-2012: columnas, tablas a todo el
  ancho y vínculo de Table 3 entre páginas físicas 8 y 9. No es un parser general.
- Párrafos mediante huecos verticales/negrita; conserva saltos de línea. No corrige
  términos médicos ni aplica dehyphenation silencioso. Secciones orientativas con
  revisión del agente y revisión humana pendiente.
- Cortes de 256, 512 y 768 tokens, con overlap 0 y 15%. El presupuesto incluye
  `passage:` y tokens especiales; cada fragmento se vuelve a contar sin truncación.
- Agrupación estructural a esos tres tamaños: no mezcla secciones, mantiene tablas
  como bloques padre y permite sobrepasar el presupuesto sólo en tablas conservadas.
- Controles de cobertura del texto seleccionado, redundancia, fragmentación, longitud,
  mezcla de secciones, tablas completas y contención de anclas de evidencia.

El código educativo de los cortes está visible en el notebook y se contrasta contra
los componentes reutilizables en `ingestion/parsing.py`, `ingestion/chunking.py` y
`evaluation/parsing.py`. La equivalencia se comprueba sobre el corpus real.

## Salidas y procedencia

Todo el texto y sus derivados se guardan localmente bajo
`data/processed/<document_id>/02_chunking/`, excluido de Git:

```text
latest.json                     Puntero a una ejecución identificada
reference_regions.png           Crops de las microreferencias
chunking_comparison.png          Comparativa calculada
runs/<run_id>/manifest.json      Corpus, configuración, entorno, hashes y estado semántico
runs/<run_id>/pages.jsonl        Texto bruto de todas las páginas con procedencia
runs/<run_id>/blocks.jsonl       Todos los bloques, incluidos referencias y avisos
runs/<run_id>/located_blocks.jsonl  Bloques del corpus seleccionado con offsets globales
runs/<run_id>/corpus.txt         Texto común para todos los chunkers
runs/<run_id>/chunks_*.jsonl     Nueve colecciones por defecto
runs/<run_id>/metrics.csv        Mediciones mecánicas de chunking
runs/<run_id>/evidence_preservation.csv  Micropruebas de contención
runs/<run_id>/parsing_reference_comparison.csv  CER/WER sobre regiones
runs/<run_id>/unresolved_objects.jsonl   Imágenes/diagrama sin transcripción estructurada
runs/<run_id>/page_audit.jsonl   Palabras fuera del mapa para revisión
```

Cada chunk guarda documento, versión, papel documental, URL y hash; huella del parser y
del corpus; estrategia/parámetros; tokenizer/revisión/hash; texto, tokens, offsets,
páginas y secciones; spans hacia bloques y figuras pendientes. El bounding box es del
bloque entero, **no de cada carácter recortado**. El ID cambia si cambia la configuración
o la identidad documental. Las ejecuciones con distinta huella no sobrescriben las
colecciones anteriores. La publicación usa staging y un bloqueo por documento; sólo un resultado completo
actualiza `latest.json`. Repetir la misma huella verifica hashes sin reescribir artefactos.
El experimento semántico opcional exporta aparte a `semantic_runs/`, sin sustituir
la colección baseline.

## Interpretar resultados sin confundir componentes

La comparación de microregiones usa `within_bbox` para exigir caracteres completamente
contenidos: `crop` puede incluir texto cuya caja sólo interseque el borde y contaminar
una transcripción de una sola línea. Los crops se contrastaron visualmente antes de
adoptar esta política; la referencia no se ajusta al texto devuelto por un parser.

La cobertura de caracteres se mide respecto al **corpus seleccionado tras parsing**.
No equivale a cobertura del PDF. El audit permite detectar omisiones, además de
cabeceras/pies excluidos. Referencias, avisos y pies de figura se conservan sin entrar en la
comparación de chunks de recuperación. Las figuras se registran como pendientes,
incluido el diagrama vectorial de la página 4: sus etiquetas sueltas no se indexan
como si fueran prosa con relaciones ya interpretadas.

Conservar un grupo de tabla en un padre largo puede mejorar la métrica de contención
y seguir siendo inviable para E5. Se muestran por separado contención en cualquier
chunk y contención en uno de hasta 512 tokens. `embedding_eligible_by_length` sólo
declara longitud compatible: no valida semántica, fidelidad o permiso de uso clínico.
Una tabla plana no garantiza preservar relación entre filas, columnas, unidades y notas.

Las diez referencias son pequeñas y seleccionadas conociendo problemas de 01. La
revisión humana sigue pendiente. Las cuatro micropruebas de evidencia no son Recall@k:
todavía no hay recuperador. Evaluar cada documento por separado; los resultados de ESUR no validan PI-RADS.

## Política específica de PI-RADS v2.1

El mapa `pirads-2.1-2019-layout.v1.json` declara las 76 páginas. `page_number` es
la página física; `page_label` es la impresa revisada: i–vii y 1–69. La etiqueta
automática de pypdf se conserva como `pdf_page_label` en los registros brutos.
Encabezados y pies se dejan fuera de las regiones. La capitular de introducción se
reúne mediante una regla optativa, sin correcciones médicas automáticas.

Se seleccionan prosa y 14 grupos de tabla (5 cuadros de evaluación y 9 páginas del
léxico). Bibliografía, portadas, créditos, ejemplos de informe cumplimentado, figuras,
el aviso de protocolos en construcción y el enlace al atlas se conservan como bloques
de referencia/metadata; no entran en la recuperación de este baseline. Las imágenes
informativas siguen pendientes de OCR/transcripción. Las 84 imágenes decorativas se
registran aparte: no deben interpretarse como 84 figuras médicas sin procesar.

Las anclas de evidencia de PI-RADS exigen página (y admiten región), evitando que un
encabezado repetido en otra tabla satisfaga la microprueba. La muestra de 10 regiones
contiene 9 textos extraíbles y un control de texto dentro de imagen; no valida todos
los párrafos ni las relaciones tabulares del manual.

## Semántico: opcional, no ejecutado por defecto

La celda usa Sentence Transformers con el mismo commit de E5, CPU, `trust_remote_code=False`
y acceso sólo local salvo que se active la descarga explícita. Instalar el extra con
`uv sync --extra embeddings`. Activar `RUN_SEMANTIC=True` y, sólo si se necesitan
descargas de pesos, `ALLOW_ENCODER_DOWNLOAD=True`. Revisar tamaño/RAM: los pesos pueden
superar 1 GB. No se activa al sincronizar dependencias ni al ejecutar el notebook por defecto.

La hipótesis consiste en proponer cortes por distancias coseno entre párrafos cortos
consecutivos, percentil 80, además de estructura y límite de tokens. Excluye párrafos
largos del cálculo de vectores y conserva las tablas. No se presenta una mejora sin
ejecutarlo y compararlo. Si no hay cortes adicionales, el resultado coincide con el
estructural y se informa así. Una activación posterior debe registrar el entorno,
el umbral y los pesos, además de las métricas.

## Siguiente paso

03 ejecutará embeddings locales y recuperación ES/EN sobre estas colecciones.
Habrá que decidir cómo tratar padres largos (hijos con encabezados/notas y referencia
al padre, un encoder de ventana mayor, o excluirlos de una comparación explícita).
El texto perdido en imágenes necesita revisión/OCR antes de poder recuperarse.
Se podrá medir recuperación sin llamar aún a un LLM de generación.
