# Arquitectura y contratos de datos

## Decisión de esta iteración

Pipeline local: catálogo → adquisición permitida → PDF inmutable + manifiesto →
inventario → regiones y párrafos → chunks con procedencia → embeddings locales →
índices NumPy → rankings y métricas. 03 ejecuta E5 y Qwen3 con sus tokenizers,
pesos y revisiones fijados. Ninguna llamada Cloud es necesaria. El cliente de
generación es independiente y se usa sólo por petición explícita. El paso 03 no
utiliza un generador; su integración en el chat se describe a continuación.

La interfaz de chat añade generación local con Qwen3 4B Instruct cuantizado,
probada en la RTX con respuesta y citas. Ollama Cloud queda como alternativa
explícita; sus intentos previos devolvieron HTTP 402. El modo actual bloquea Cloud,
libera el encoder antes de generar y descarga el generador al terminar.
Véase [chat.md](chat.md). No hay grafo implementado.

Python 3.12 limita fricción de wheels científicos en Windows. `uv.lock` fija versiones;
`uv sync --locked` instala el conjunto por defecto sin pesos de modelos. El extra
embeddings es necesario para 03; en Windows fija PyTorch con CUDA 12.6. Los pesos
se descargan con un script separado. Las funciones de recuperación y evaluación
están implementadas y probadas; los grafos siguen pendientes.

## Notebook frente a librería

Los cuadernos contienen los bucles de exploración, métricas, detección de cabeceras,
comparación de parsers y exportación. El paquete sólo concentra configuración,
localización del proyecto, catálogo, hash, descarga/incorporación local, parsing, chunking, publicación y HTTP. Los
algoritmos de corte aparecen también implementados paso a paso en 02. 03 muestra
forward, pooling, similitud y métricas, contrastados con las funciones reutilizables.
Así se puede inspeccionar
la lógica educativa y reutilizar infraestructura sin copiar credenciales o rutas.

## Contrato preparado para 02

El notebook 01 exporta en `data/processed/<document_id>/`:

- `page_profile.csv`: página física (1-based), etiqueta PDF, tamaños, caracteres,
  palabras, imágenes incrustadas y candidatos a tabla. Son indicadores, no exactitud.
- `pages.jsonl`: `document_id`, `document_version`, `document_role`, `source_url`,
  `source_sha256`, `page_number`, `page_label`, `section=null`, `text`, `parser`,
  `parser_version`, `created_at`. `section=null` evita inventar segmentación.
- `exploration.json`: entorno, documento y mediciones agregadas reproducibles.

El catálogo fija `active_document_id` y las rutas de mapa, referencias y ejemplos
por documento. PI-RADS v2.1 es activo desde el 2026-11-13; ESUR sigue separado.
La CLI `import` conserva procedencia local y `process` publica las nueve colecciones
por huella, con bloqueo, staging y hashes. 02 contrasta los algoritmos educativos
contra los IDs de ese mismo resultado completo.

La página física siempre identifica una página del archivo. La etiqueta interna del
PDF puede no coincidir con la página impresa; 01 permite contrastarlas visualmente.
No equiparar etiquetas automáticas con numeración editorial sin revisión.

El contrato inicial de chunk se implementa y amplía en 02. La descripción completa,
campos reales y salidas están en [chunking.md](chunking.md). El siguiente ejemplo es
orientativo; no es un resultado de ejecución:

```json
{
  "chunk_id": "hash(document_id, version, sha256, strategy, span)",
  "document_id": "esur-2012",
  "document_version": "ESUR-2012 / antecedente PI-RADS v1",
  "source_url": "URL oficial",
  "source_sha256": "hash del archivo",
  "pages": [1],
  "section_path": [],
  "text": "texto extraído, sólo local",
  "strategy": "fixed|structural|semantic",
  "parser": "nombre y versión",
  "bbox": null,
  "review_status": "unreviewed"
}
```

No copiar este ejemplo como resultado. Conservar tablas y sus títulos/notas en
unidades vinculadas; las figuras requieren inspección o un paso multimodal explícito.
No generar descripciones de imágenes desde sus nombres. Los índices de 03 tienen
huella de corpus, chunker, modelo/revisión de embeddings, precisión y normalización.

## Contrato de recuperación de 03

`data/runs/<document_id>/03_embeddings/latest.json` referencia un informe completo
y el índice correspondiente. Cada matriz `embeddings.npy` tiene el orden de su
`passages.jsonl`. Se guardan consultas y anotaciones aparte. Los manifiestos incluyen
hashes de artefactos, corpus, piloto, modelos, código, lockfile y entorno.
La publicación usa staging y bloqueo; sólo actualiza el puntero al completar ambos
destinos. La caché de vectores valida hash, forma y normalización antes de reutilizarse.

La recuperación es exacta: producto matricial de vectores L2 normalizados y orden
estable de scores. Los padres largos producen hijos con offsets y referencia al
padre. Los modelos se cargan de uno en uno, en float16, CUDA, batch 1. El pooling
es media enmascarada para E5 y último token válido para Qwen3. Los vectores se
persisten en float32. Las métricas y umbrales se mantienen separados de cualquier
generador; la calibración acepta exclusivamente registros de desarrollo.
Véanse [resultados y límites de 03](embeddings.md).

## Contrato LLM

El modo predeterminado actual usa `http://127.0.0.1:11434`, modelo local Qwen3 4B
Instruct y `OLLAMA_NO_CLOUD=1`. HTTP sólo se permite para loopback; no se envía
Authorization al endpoint local. El flag también bloquea endpoints remotos en
el cliente. Contexto 8192, salida máxima 512, batch 128, dos hilos CPU y keep_alive=0.
Se libera el encoder antes de generar y se valida el presupuesto con el tokenizer
Qwen3 Instruct fijado. El contrato remoto siguiente queda como alternativa explícita.

API nativa `POST https://ollama.com/api/chat`, Bearer, `stream=false`. El nombre del
modelo procede de `GET /api/tags`; no se fija uno que pueda retirarse. El cliente usa
`options.temperature` y `options.num_predict`, con soporte efectivo dependiente del
modelo. Fuentes: [Cloud](https://docs.ollama.com/cloud),
[chat](https://docs.ollama.com/api/chat), [errores](https://docs.ollama.com/api/errors).

Reintentar sólo transporte/timeout y 429/500/502/503/504; detener 400/401/403/404.
Backoff exponencial con jitter, máximo configurable, y `Retry-After` en segundos o
fecha. Si el servidor exige una espera mayor que el máximo, detener para revisión de
cuota. No seguir redirecciones con credenciales. La URL configurable debe ser un origen
HTTPS de confianza; al cambiarla, la clave se envía a ese origen.

La latencia incluye esperas e intentos. Los contadores de tokens y nanosegundos del
servidor se conservan sólo si llegan. No hay estimación monetaria sin tarifa verificable.
El logger de la librería registra únicamente métricas, nunca prompts, respuestas o
claves. No activar logs HTTP detallados en sesiones con credenciales.

`llm-smoke` imprime la respuesta mínima y las métricas, pero no la persiste. Un futuro
runner guardará sólo métricas en `data/runs/`; almacenar respuestas/contexto será una
decisión explícita de cada experimento. Los tests HTTP usan `httpx.MockTransport`;
esa evidencia prueba el cliente, no la disponibilidad o calidad del servicio.
