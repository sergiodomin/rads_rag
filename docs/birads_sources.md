# Corpus BI-RADS

## Fuentes seleccionadas

Verificación: 2026-11-15. La edición del sistema y la fecha de publicación de un
artículo son campos distintos; una revisión de 2026 puede describir BI-RADS v2025.

| ID | Fuente | Papel | Disponibilidad |
|---|---|---|---|
| `birads-v2025-manual` | ACR BI-RADS v2025 Manual | Manual oficial | Referencia catalogada; copia autorizada pendiente |
| `birads-v2025-review-2026` | Kaplan E, Aydemir AB. Diagnostics 2026;16(13):2135 | Revisión narrativa secundaria de v2025 | PDF/XML incorporados e índice estructural preparado |

ACR identifica [v2025 como edición actual](https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS)
y ofrece el manual mediante compra y licencias de software. El catálogo no presume
permiso de procesamiento a partir de acceso público a la página o de una compra.

La revisión [How Does ACR BI-RADS v2025 Change the Radiologist's Approach?](https://doi.org/10.3390/diagnostics16132135)
se publicó el 2026-07-07. La editorial y los metadatos del repositorio PMC confirman
CC BY 4.0. Atribución: Ela Kaplan y Ahmet Burak Aydemir, Diagnostics 16(13), 2135,
DOI `10.3390/diagnostics16132135`. No sustituye al manual ACR ni se etiqueta como
guía normativa. Revisar créditos particulares de figuras antes de reutilizarlas.

## Artefactos locales

En `data/raw/birads-v2025-review-2026/`, fuera de Git:

- `review.pdf`: 31 páginas; 1.470.545 bytes.
- `review.source.json`: procedencia, fecha de adquisición e identidad del PDF.
- `review.xml`: fuente editorial estructurada JATS; contiene 6 tablas y 2 figuras
  referenciadas. Las imágenes originales no se han descargado ni interpretado.
- `repository_metadata.json`: metadatos oficiales de PMC, incluida licencia.
- `structured_source.json`: hashes del PDF/XML y recuentos en la adquisición.
  El estado de indexación vigente se consulta en el puntero de `03_embeddings`;
  el registro original de adquisición no se reescribe.

La adquisición se verificó contra los MD5 publicados por PMC y se fijaron SHA-256
en `data/catalog/documents.v1.json`. El PDF se puede reproducir mediante
`uv run radiology-rag download birads-v2025-review-2026`; si ya existe no se
sobrescribe. `uv run radiology-rag verify` comprueba la identidad incorporada.

## Incorporación a la recuperación

La integración usa `ingestion/jats.py`: agrupa párrafos contiguos de la misma
sección y genera pasajes por fila tabular, conservando encabezados, subsecciones
y notas. Expande `rowspan`/`colspan` y reconoce los encabezados internos en negrita
de esta publicación. Es una regla del perfil editorial, con revisión humana
pendiente, no un parser universal de tablas.

El resultado contiene 234 pasajes: 40 de prosa, 189 de filas y 5 de notas tabulares.
Las notas completas permanecen en el contexto de cada fila; para codificar la fila
se excluye su repetición y se indexan también como pasajes propios. Esto evita que
una nota común domine la similitud de todas las filas. Dos pasajes son hijos de
una fila larga; al recuperarlos se entrega la fila completa con su contexto.
Se deduplican padres antes de completar hasta cuatro resultados.

Para recuperar categorías se derivan `assessment_categories`, encabezados y
celdas a partir del XML verificado. También se enriquecen de esa forma los índices
existentes al cargarlos, sin modificar sus archivos ni recalcular embeddings.
`retrieval/hybrid.py` combina BM25 y coseno mediante RRF y prioriza coincidencias
explícitas de sistema/categoría. Un número de tabla o un año no se trata como categoría.
La pregunta original y su normalización quedan separadas en el contexto y métricas.

Las seis tablas completas, celdas y XML original se conservan en `objects.json`.
Las dos figuras quedan registradas como no interpretadas y no se indexan.
Las citas BI-RADS usan sección, ID XML y fila; no se inventa una correspondencia
con páginas PDF. PI-RADS conserva sus páginas físicas y etiquetas impresas.

Reproducción:

```powershell
uv run python scripts/prepare_birads_source.py
uv run --extra embeddings python scripts/prepare_birads_index.py
uv run --locked --extra embeddings radiology-rag chat --open
```

El constructor verifica PDF/XML, usa CUDA con batch 1 y modelos secuenciales,
publica artefactos completos y después registra `chat_document_ids`. El chat
rechaza mezclar revisiones de encoder entre documentos. El piloto histórico de
PI-RADS no constituye una evaluación BI-RADS ni una evaluación multi-corpus.

BI-RADS sólo dispone de colección estructural. La interfaz permite esa colección
cuando se consultan todas las fuentes; los cortes fijos siguen disponibles para
PI-RADS por separado. No se fuerza a que una búsqueda conjunta incluya ambos
sistemas: los candidatos se ordenan por similitud en el mismo espacio vectorial.

La recuperación multi-corpus debe conservar `system`, `document_id`, versión del
sistema, fecha de publicación, papel documental, modalidad, DOI y hash. Las citas
deben identificar el documento concreto. Una revisión que compara 2013 y v2025
también requiere atribuir cada afirmación a la edición que describe.

## Almacenamiento y compatibilidad

Se mantiene el almacenamiento actual JSONL + NumPy. La migración a PostgreSQL o
pgvector queda aplazada; la incorporación de BI-RADS no depende de esa decisión.
Mantener identidad documental, metadatos consultables y procedencia del original.

Usar el mismo encoder/revisión al ordenar candidatos de varios sistemas; los
vectores Qwen de 1024 dimensiones y E5 de 768 requieren espacios de búsqueda
separados. Evaluar la recuperación multi-corpus antes de cambiar el backend.
