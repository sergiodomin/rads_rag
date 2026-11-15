# Laboratorio de evaluación

Vista local en **http://127.0.0.1:8765/lab**, accesible desde el chat. Compara
recuperación, contexto y generación conservando las entradas de cada experimento.
No utiliza un juez LLM ni considera una cita válida prueba de fidelidad.

## Preparación

Con ambos índices y el generador local preparados:

```powershell
uv run python scripts/prepare_lab_dataset.py
uv run --locked --extra embeddings radiology-rag chat
```

El script no carga pesos ni hace inferencia. Crea
`data/evaluations/lab.v1.json` y conserva las revisiones si ya existe. El seed
publicable contiene preguntas y offsets/hashes; las transcripciones se extraen de
los documentos locales. Si cambia el parsing, falla antes de reutilizar offsets.

## Casos y referencias

- 48 consultas: 24 grupos en español e inglés, 40 de desarrollo y 8 de test.
- Preguntas de PI-RADS, categorías y cambios de BI-RADS, comparaciones, tablas,
  ausencia de información y dos situaciones conversacionales con historial fijo.
- Traducciones del mismo caso permanecen en la misma partición. Los ejemplos del
  antiguo piloto y fallos conocidos quedan en desarrollo. El antiguo control de
  mama fuera de dominio se sustituye, pues ahora hay un corpus BI-RADS.
- Cada referencia vincula documento, versión, papel documental, hashes y página
  física/ancla o elemento XML/fila. Las respuestas iniciales son extractos ingleses
  propuestos automáticamente, **no respuestas expertas ni referencias validadas**.

En «Inspeccionar y revisar referencia» se puede corregir la respuesta, añadir
notas y marcar la revisión documental. Las anclas y la suficiencia también deben
comprobarse. Para modificar evidencia, editar el JSON local conservando el esquema;
se valida su identidad y resolución antes de ejecutar. Una revisión documental no
equivale a validación por un radiólogo. Historial de revisiones en
`data/evaluations/lab_history/`.

Reservar test para una configuración fijada; no usarlo para ajustar consultas,
reglas ni parámetros. La separación inicial tiene sólo cuatro grupos nuevos de
BI-RADS en test: no es una muestra representativa de todos los RADS.

## Condiciones

| Condición | Evidencia | Ejecución |
|---|---|---|
| Sólo recuperación | Buscador real | Sin generación; mide cobertura de anclas |
| Sin contexto | Ningún documento | Control de conocimiento previo con instrucciones propias |
| Contexto de referencia | Extractos anotados previamente | Omite el buscador; permite investigar errores del generador |
| Contexto recuperado | Fragmentos del buscador | Generación libre con el prompt documental |
| Baseline de producto textual | Fragmentos del buscador | Qwen3 4B y campos de alcance verificados; no es el chat multimodal común |

Los controles del laboratorio mantienen el baseline textual. El pipeline común del
chat registra su contexto mixto y sus salidas en `output/unified/runs/`; no se ha
incorporado todavía como condición A/B de esta pantalla.

Las referencias en borrador pueden ser incompletas. El control de referencia sólo
constituye «evidencia correcta y suficiente» tras revisarla. Los controles de
referencia y RAG comparten las instrucciones documentales; el de conocimiento
previo tiene un prompt distinto, registrado en la traza. Los historiales son
fixtures, no conversaciones generadas de extremo a extremo. Las comparaciones
explícitas omiten el historial como en el chat.

Las respuestas con campos verificados se identifican mediante `answer_mode` y
`generation_executed=false`. No contarlas como éxito del modelo generativo.

## Recuperadores y generadores

Se pueden contrastar coseno, BM25 y el híbrido actual con RRF y reglas de categoría.
Además, se puede activar [reranking local](reranking.md), fijar un pool de hasta
20/40 candidatos y contrastar cuerpo de contexto padre/pasaje con sus metadatos.
BM25 no carga el encoder. Se conservan filtros, planificación por sistemas,
deduplicación por padre y hasta cuatro fuentes; por tanto son ablaciones del
ranking dentro del plan actual, no baselines sin planificación. La segmentación
fija sólo está disponible para PI-RADS; seleccionar estructural al incluir BI-RADS.

Inicialmente hay **un generador registrado**, Qwen3 4B Instruct Q4_K_M. No se ha
comparado aún contra un segundo modelo. `data/catalog/lab_generators.v1.json`
admite hasta dos perfiles seleccionados por lote. Cada perfil necesita modelo
local instalado, digest y tokenizer propio con hashes, compatible con el formato
ChatML implementado. Otros formatos requieren implementar su presupuesto nativo;
no reutilizar el tokenizer actual sin comprobar compatibilidad. No hay descargas
automáticas ni llamadas a Cloud.

Las ejecuciones son secuenciales, comparten bloqueo con el chat y limitan cada lote
a 12 llamadas de generación como máximo. El encoder se descarga antes del LLM;
Ollama usa `keep_alive=0`. Detener solicita cancelación entre tareas: no interrumpe
una llamada nativa en curso. Un reinicio del servidor marca las ejecuciones
inacabadas como interrumpidas; no las reanuda automáticamente.

## Métricas y revisión

- `evidence_recall`: proporción de unidades anotadas encontradas entre las fuentes
  finales, validando hashes, localización y ancla. Duplicados no suman cobertura.
- `mrr`: inverso de la posición de la primera fuente que cubre evidencia anotada.
- `novel_evidence_ndcg`: descuento por posición de fuentes que añaden evidencia
  nueva. Es una medida propia de cobertura descontada, **no nDCG estándar** con
  juicios exhaustivos de relevancia por pasaje. No compara exactitud clínica.
- Controles sin evidencia positiva: recall/MRR nulos, excluidos de sus medias.
  Una falta de anotación no demuestra que no exista evidencia en el corpus.
- IDs de citas válidos, abstención, truncamiento, llamadas reales, tiempos y tokens.
  El resumen informa tamaños de muestra y desgloses por idioma, tipo y partición.
- Fidelidad, corrección y apoyo real de las citas permanecen sin puntuar hasta
  revisión humana. «Salida bruta» y «tras controles» se guardan por separado para
  que una abstención del filtro no oculte un error del modelo.

La revisión permite registrar afirmaciones, juicio y fuentes que las apoyan o
contradicen. `S1` identifica contexto enviado al modelo; `R1`, evidencia de
referencia. La corrección puede contrastarse con R; la fidelidad exige comprobar
el contexto S realmente recibido. Se conserva la valoración de la salida bruta.
Los resúmenes muestran recuentos humanos sin fabricar una exactitud clínica global.

## Trazabilidad y comparación

Cada lote conserva en `output/lab/runs/<id>/`:

```text
snapshot.json       Dataset completo, configuración, perfiles, digest y hashes
run.json            Estado, progreso y resumen automático del lote
records/0001.json   Consulta, ranking, contexto, mensajes, presupuesto y salidas
reviews/<id>.json   Valoraciones humanas posteriores, separadas del experimento
```

La interfaz permite elegir ejecuciones A/B y descargar JSON con snapshot, resultados
y revisiones actuales. Una revisión posterior del dataset no modifica snapshots
anteriores. Los archivos son locales y están ignorados por Git; contienen texto
documental y consultas.

La latencia incluye carga y ejecución; no se purgan las cachés del sistema. No se
muestrea todavía el pico de RAM/VRAM ni se controlan todas las fuentes de variación
del runtime. No interpretar una única pasada como benchmark de rendimiento.

Primer procedimiento: revisar un pequeño conjunto de desarrollo, ejecutar sólo
recuperación con dos configuraciones y examinar fallos. Después, en los mismos
casos, contrastar los tres controles generativos. Clasificar errores antes de
incorporar reranking o técnicas nuevas. La comparación entre generadores, una
revisión experta y un test final independiente siguen pendientes.
