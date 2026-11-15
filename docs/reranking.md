# Reranking y selección del contexto

Disponible en `/lab` y activado por defecto en el chat. «Reordenar fuentes» permite
seleccionar Qwen3-Reranker-0.6B o desactivarlo. El perfil del chat está en
`data/catalog/chat_retrieval.v1.json`: hasta 20 candidatos, cuerpo del padre y
cuatro fuentes finales. No se modifican los índices ni se recalculan embeddings.

## Pipeline

```text
Pregunta → encoder/BM25 → hasta 20 o 40 candidatos
         → Qwen3-Reranker-0.6B opcional
         → selección y deduplicación → hasta 4 fuentes → generador opcional
```

El encoder representa pregunta y documentos por separado. El reranker lee juntos
pregunta y pasaje para puntuar su relevancia. No genera la respuesta, no produce
embeddings nuevos y no certifica la veracidad de un pasaje.

Se usa [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), licencia
Apache-2.0, revisión `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
Su patrón de scoring compara los logits de `yes` y `no` tras la plantilla indicada
por el autor. Se normalizan esos dos logits con softmax: el resultado entre 0 y 1
es una puntuación de ranking, **no una probabilidad clínica ni confianza calibrada**.

Catálogo, tamaños y SHA-256 en `data/catalog/reranker_models.v1.json`. Pesos locales
en `data/raw/models/`, ignorados por Git. Se comprueban los hashes antes de cargar;
no se permite código remoto ni descargas durante la inferencia.

## Preparación

```powershell
uv run python scripts/prepare_reranker.py
uv run --extra embeddings python scripts/check_reranker.py
uv run --locked --extra embeddings radiology-rag chat
```

La descarga ocupa aproximadamente 1,2 GB. La prueba sintética usa una consulta
española y dos pasajes ingleses; sólo comprueba carga, scoring y orden básico.
No es una evaluación radiológica.

## Límites de ejecución

- CUDA, float16 y batch de un par; CPU con dos hilos.
- Encoder, reranker y generador se cargan por turnos. Se liberan también las
  referencias locales al encoder antes del reranking.
- Presupuesto operativo de **2048 tokens por par**, contado con el tokenizer propio
  del reranker. No corresponde al tamaño de chunk de E5 ni al contexto del LLM.
- Consulta y plantilla completas. Si no caben, se rechaza la ejecución. Si el
  documento supera el espacio restante, se recorta su final y se registra el número
  de tokens originales, conservados y si hubo truncamiento.
- El forward solicita sólo los logits de la última posición, sin generar tokens
  ni conservar una caché KV entre pares.
- Errores de carga/scoring se registran como errores. No hay fallback silencioso
  que aparente haber ejecutado el reranker.

Se registran carga/verificación, scoring, pares, truncamientos, memoria CUDA máxima
asignada/reservada por PyTorch y RSS del proceso al terminar el scoring. RSS es una
medición puntual; las métricas CUDA no incluyen todos los procesos ni toda la
memoria del controlador. No son medidas del pico total del equipo.

## Selección del contexto

Los candidatos provienen del mismo recuperador, filtros y planificación que el
baseline del experimento. El máximo es 20 o 40; puede haber menos, especialmente
cuando el plan limita una comparación general a las introducciones.

El reranker reordena ese pool y después se deduplican los padres. La selección
conserva las restricciones del producto: dos plazas por sistema en comparaciones,
anclas de alcance para comparaciones específicas sin categoría y prioridad de
coincidencias de categoría en el recuperador híbrido. El ranking aprendido y las
restricciones de selección son pasos distintos y se ven por separado en la traza.
No es una ablación de reranking sin reglas. El reranker no puede recuperar evidencia
que ya quedó fuera del pool.

La opción «Cuerpo del contexto» permite usar el texto del padre o el del pasaje
indexado. Los metadatos de procedencia y los pares columna/valor de tablas se
conservan en ambos casos. Por tanto, cambiar a pasaje no elimina toda la información
estructurada del padre: es una ablación del cuerpo textual, no del contexto completo.
La expansión al padre tampoco crea nuevas coordenadas de página.

Se mantiene el máximo de cuatro fuentes y el presupuesto global del generador.
Igualar número de fuentes no iguala tokens: consultar `context_characters` y, en
experimentos con generación, el conteo real del tokenizer del LLM.

## Experimento A/B

En `/lab`, seleccionar desarrollo y «Sólo recuperación»:

1. Fijar encoder, recuperador, segmentación, candidatos y cuerpo del contexto.
2. Ejecutar con «Sin reranking».
3. Repetir los mismos casos con Qwen3-Reranker-0.6B.
4. Comparar los dos lotes en A/B: tabla de orden inicial, orden aprendido, puntuación
   y posición seleccionada como fuente final. Abrir texto y procedencia cuando
   interese comprobar un cambio.

También se puede ejecutar una muestra fija de seis consultas de desarrollo, tres
grupos bilingües, sin generar respuestas:

```powershell
uv run python scripts/compare_reranking.py
```

El script comprueba que dataset, índices, configuración y candidatos/textos coinciden
entre los dos lotes. Guarda el informe en
`output/lab/reranking_comparison.json`; los lotes completos quedan accesibles en la
web. No modifica referencias ni usa test. Para probar 40 candidatos, ejecutar con
`--candidate-count 40` como otro experimento.

Las métricas distinguen cobertura del pool, selección inicial y selección final.
Si falta evidencia en el pool, investigar la primera recuperación. Si está en el
pool y desaparece al seleccionar, investigar el ranking, la deduplicación o las
restricciones. Con referencias en borrador, los resultados son exploratorios.
No deducir calidad de respuesta de una mejora en recall.

Las latencias incluyen cargas y cachés no controladas. La muestra no permite elegir
un ganador general ni justificar un despliegue automático en el chat. Siguen
pendientes referencias revisadas, evaluación más amplia por tipo/idioma y aceptación
de una configuración antes de consultar test. La variación del tamaño de chunks,
overlap, reescritura de consultas y presupuesto de contexto son experimentos futuros.

## Primer resultado exploratorio

En la primera ejecución de la muestra fija de seis consultas (tres grupos ES/EN),
con 20 candidatos máximos y cuerpo del padre, el recall medio de evidencia anotada
fue **0,5833 sin reranking** y **0,5000 con reranking**. El script verificó igualdad
de candidatos y textos entre lotes. No hubo llamadas al generador.

Las referencias eran borradores y la muestra es pequeña: no demuestra inferioridad
general del modelo, pero tampoco justifica sustituir la selección del chat.
El chat permite utilizarlo y desactivarlo; su activación no implica mejora validada.
El informe completo y los IDs de ambos lotes se guardan localmente en
`output/lab/reranking_comparison.json`.
