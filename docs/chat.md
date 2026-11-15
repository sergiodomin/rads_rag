# Chat documental local

Backend HTTP de Python e interfaz HTML/CSS/JavaScript. Escucha en
`127.0.0.1:8765`. Embeddings y generación se ejecutan localmente con CUDA y Ollama.

El chat principal usa ahora recuperación fusionada y Qwen3-VL 2B Instruct como
generador común para texto, tablas e imágenes. [Pipeline actual](multimodal.md).
Las opciones de Qwen3 4B y campos verificados descritas abajo corresponden al
baseline textual conservado en el laboratorio y en `ChatEngine.respond`.

## Preparación y ejecución

Seguir la [instalación del proyecto](../README.md). Para incorporar la segunda
familia, con los modelos de embeddings preparados:

```powershell
uv run python scripts/prepare_birads_source.py
uv run --extra embeddings python scripts/prepare_birads_index.py
uv run --locked --extra embeddings radiology-rag chat --open
```

`Ctrl+C` cierra el servidor iniciado en esa terminal. Se puede omitir `--open` y
abrir manualmente `http://127.0.0.1:8765`; `--port` permite cambiar el puerto.
Después de preparar nuevos índices hay que reiniciar el servidor para cargarlos.

## Fuentes y segmentación

| Sistema | Documento disponible | Segmentación |
|---|---|---|
| PI-RADS | Manual oficial v2.1, 2019 | Fija 512/15 % o estructural, 50/56 pasajes |
| BI-RADS | Kaplan y Aydemir, revisión de 2026 sobre v2025 | Estructural JATS, 234 pasajes |

La revisión BI-RADS es una fuente secundaria. El manual oficial BI-RADS v2025 no
está incorporado. Sus figuras no están interpretadas; el texto de una leyenda no
se presenta como comprensión visual de la figura.

La interfaz permite seleccionar PI-RADS, BI-RADS o todas las fuentes. Para consultar
ambas se usa segmentación estructural, disponible en los dos documentos. Se buscan
los mejores candidatos mediante búsqueda híbrida en el espacio del encoder seleccionado; no se fuerza
a incluir un documento irrelevante para completar una cuota por sistema.

Cuando la pregunta nombra explícitamente ambos sistemas, se planifican búsquedas
independientes y se reservan hasta dos padres distintos por sistema. Las consultas
generales de diferencias usan las secciones de introducción/alcance cuando existen;
las preguntas específicas conservan los candidatos generales y tabulares. Si el
filtro excluye un sistema solicitado, la API pide seleccionar todas las fuentes.
Estas comparaciones explícitas no añaden la pregunta anterior al embedding ni
envían el historial previo al generador, evitando arrastrar categorías de otro turno.
Los fragmentos separados de ambos documentos son evidencia válida para sintetizar
una comparación; no es necesario encontrar un párrafo que ya los compare.
Las comparaciones generales usan dos puntos breves, con citas por sistema y sin
cronología ni detalles clínicos no solicitados.
El alcance se presenta mediante `generation/scope.py`, a partir de perfiles
revisados de `data/catalog/system_profiles.v1.json`. Cada órgano/modalidad debe
encontrar evidencia en los fragmentos recuperados y corresponder al documento y
hashes del perfil. La respuesta se construye con esos campos, sin redacción del
LLM, y se identifica como «Comparación estructurada de fuentes». Sólo enumera
modalidades respaldadas por los fragmentos; no pretende ser una lista exhaustiva.
Si faltan campos o cambia la identidad del documento, se abstiene sin pedir al
LLM que rellene los datos. Las consultas de detalle conservan generación local.

Para cualquier comparación sin categorías explícitas se conserva al menos un
pasaje de introducción/alcance por sistema, incluso si una variante de la pregunta
no se reconoce como comparación general. Las fórmulas de petición habituales
(«dime», «explícame», «por favor») no se tratan como un tema técnico.

El filtro de sistema se aplica antes del ranking. Se normalizan variantes conocidas
como `pirads`, `pi rads`, `PI-RADS`, `birads`, `bi rads` o `birdas`, conservando
la pregunta original. También se admiten mayúsculas, guiones Unicode, espacios,
guion bajo y categoría pegada al nombre. El registro `SYSTEM_ALIASES` centraliza
las familias disponibles; no corrige palabras médicas arbitrarias ni mezcla sistemas.
Se combinan búsqueda
por coseno y BM25 mediante RRF (constante 60, hasta 50 candidatos por recuperador).
Por defecto, Qwen3-Reranker-0.6B reordena un pool de hasta 20 candidatos antes
de seleccionar las cuatro fuentes. «Reordenar fuentes» permite desactivarlo por
consulta; también se aplica en modo sólo búsqueda. El perfil predeterminado está
en `data/catalog/chat_retrieval.v1.json`. El encoder se libera antes del reranker
y éste antes del generador. [Implementación y límites](reranking.md).
Las categorías explícitas se extraen de celdas bajo encabezados de categoría, no
de números de tabla o años. Si existe una fila del sistema/categoría solicitados,
se prioriza y se excluyen del grupo tabular las otras categorías de ese sistema.
La prosa y otras tablas siguen disponibles como contexto. PI-RADS no tiene aún
esa anotación de celdas; allí se combina recuperación léxica y vectorial.

Se deduplican padres y se entregan hasta
cuatro pasajes. Para las filas BI-RADS divididas por longitud se entrega al LLM la
fila completa con encabezados y notas; el presupuesto del generador se comprueba
después de esa expansión. Las tablas completas se conservan en los artefactos.
El prompt incluye también pares columna-valor, sistema, versión y categoría.
Las notas y los cambios entre versiones se distinguen de las definiciones.
La interfaz muestra las correcciones de nombres y etiqueta coincidencias de categoría;
el coseno mostrado es una medida separada del orden híbrido.

Las citas PI-RADS incluyen páginas físicas e impresas. BI-RADS utiliza sección,
ID XML y fila tabular, con enlace al artículo original. No se calcula una página
PDF por aproximación. Ambos PDF también pueden abrirse desde la biblioteca lateral.

Cambiar el sistema limpia el historial utilizado para nuevas preguntas. Las
respuestas anteriores conservan sus propias fuentes y citas.

## Modelos y memoria

Encoders disponibles: E5 multilingual base y Qwen3 Embedding 0.6B. Cada uno usa
sus pesos, tokenizer, pooling e instrucciones fijados en el catálogo. Un índice
con modelos o revisiones incompatibles entre documentos se rechaza al cargarlo.

El servidor mantiene un encoder a la vez, con float16, batch 1 y dos hilos CPU.
Antes de cargarlo comprueba CUDA y al menos 2 GiB de RAM libre. Atiende una
consulta de recuperación/generación cada vez. Libera el encoder antes de generar.

Generador local: `qwen3:4b-instruct-2507-q4_K_M`, contexto 8192 y hasta 512 tokens
de salida. El tokenizer fijado cuenta el prompt con margen y reserva de salida;
el exceso se rechaza sin truncación silenciosa. Ollama recibe `keep_alive=0` para
descargar el modelo al terminar.

## API

- `GET /api/status`: documentos, colecciones, modelos y estado de ejecución.
- `POST /api/search`: recuperación sin generador.
- `POST /api/chat`: recuperación y respuesta con citas.
- `GET /documents/<document_id>.pdf`: sólo documentos registrados y cargados.
- `GET /document.pdf`: alias compatible con las citas históricas de PI-RADS.

Ejemplo de cuerpo JSON:

```json
{
  "message": "¿Qué cambios describe la revisión sobre BI-RADS v2025?",
  "system": "BI-RADS",
  "encoder": "qwen3-0.6b",
  "collection": "structural_512",
  "history": []
}
```

Valores de `system`: `PI-RADS`, `BI-RADS`, `all`. Por compatibilidad, la API sin
filtro explícito usa PI-RADS y colección fija; la interfaz envía siempre el filtro
y arranca en todas las fuentes con colección estructural.

## Seguridad y límites

El servidor verifica hashes, procedencia del PDF/XML y matrices antes de usarlas.
Sirve rutas explícitas; no expone el directorio del proyecto. Rechaza orígenes
externos, Host no permitido, cuerpos excesivos y parámetros no válidos. La clave
del proveedor permanece en el servidor; las fuentes se renderizan como texto.

El prompt distingue sistema, versión y papel documental. Cada afirmación debe
citar fuentes del turno actual; una comparación necesita evidencia de los sistemas
solicitados. El historial conversacional no constituye evidencia documental.
La validación automática de IDs de cita no comprueba fidelidad semántica.
Se acompaña la evidencia de un glosario anatómico mínimo inglés/español para
`prostate`/próstata, `breast`/mama y `penis`/pene, sólo cuando el término aparece
en los fragmentos o títulos. Un control rechaza la introducción de esos órganos
sin respaldo textual. Es un control limitado de vocabulario, no una validación
general de anatomía o exactitud clínica. Referencias terminológicas:
[próstata](https://www.cancer.gov/espanol/publicaciones/diccionarios/diccionario-cancer/def/prostata)
y [mama](https://www.cancer.gov/espanol/publicaciones/diccionarios/diccionario-cancer/def/mama), NCI.

Con `OLLAMA_NO_CLOUD=1` se bloquean proveedores remotos. Las consultas de búsqueda
no llaman al generador. El modo remoto opcional requiere configuración explícita;
envía pregunta, historial y fragmentos al proveedor y puede consumir cuota.

Diagnóstico: `scripts/check_ollama.py` hace una petición mínima al proveedor
configurado; `--list-only` evita generar. `scripts/check_local_rag.py` comprueba
una pregunta PI-RADS con el generador local. Esas pruebas funcionales no constituyen
evaluación clínica ni una medida de calidad multi-corpus.
