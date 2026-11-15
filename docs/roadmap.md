# Roadmap: producto y laboratorio RAG

Actualizado: 2026-11-15.

## Objetivos

1. Construir un asistente documental RADS reproducible y demostrable, con fuentes
   trazables y límites visibles.
2. Comprender y comparar recuperación híbrida, reranking, generación fundamentada,
   RAG multimodal, CAG, grafos y recuperación iterativa mediante experimentos propios.

Cada ampliación debe tener una hipótesis, un baseline, un experimento y un resultado
medido. El roadmap es la referencia de prioridades; las técnicas futuras se
presentan como propuestas, no como capacidades implementadas.

## Punto de partida

| Componente | Estado |
|---|---|
| Corpus | PI-RADS v2.1, manual oficial de 2019; revisión abierta de 2026 sobre BI-RADS v2025 |
| Identidad documental | Catálogo, versiones, papel documental, procedencia y hashes |
| Parsing | Regiones PDF para PI-RADS; XML JATS y tablas estructuradas para BI-RADS |
| Segmentación | PI-RADS: fija y estructural; BI-RADS: secciones, filas y notas |
| Embeddings | E5 multilingual base y Qwen3 Embedding 0.6B, con revisiones fijadas |
| Recuperación | Coseno + BM25, RRF, filtros y coincidencias de categorías tabulares |
| Almacenamiento | JSONL + NumPy; migración a pgvector aplazada |
| Generación | Qwen3-VL 2B Instruct como generador común; Qwen3 4B conservado como baseline del laboratorio |
| Comparación general | Generación con el contexto común; campos verificados sin LLM conservados en el baseline |
| Interfaz | Chat local, selección de fuentes, citas y apertura de documentos |
| Evaluación | Pruebas de software y regresiones; piloto PI-RADS pequeño, con revisión humana pendiente |
| Figuras | Registradas, sin interpretación visual integrada |
| Notebooks | 00–03 implementados; etapas experimentales posteriores pendientes |

La comparación estructurada y la respuesta generativa son métodos diferentes y
deben identificarse y evaluarse por separado. Una prueba de software superada o
un ID de cita válido no demuestran fidelidad documental.

Resultados y límites actuales: [validación](validation.md),
[embeddings](embeddings.md), [corpus BI-RADS](birads_sources.md).

## Dos modos de aplicación

### Consulta

Interfaz estable con elección de fuentes, identidad y versión de cada documento,
respuestas breves, evidencia consultable y abstención comprensible. Mostrar si la
respuesta procede de campos estructurados o de generación libre. Mantener visibles
las diferencias entre manual oficial, revisión secundaria y documento histórico.

### Laboratorio

Comparación de configuraciones y trazas de cada consulta:

- Pregunta original, normalización, historial utilizado y plan de búsqueda.
- Candidatos vectoriales, léxicos, fusión y reranking cuando esté disponible.
- Fragmentos realmente enviados al generador, expansiones y presupuesto de contexto.
- Respuesta, citas, evaluación por afirmación y causa de abstención.
- Tiempos por fase, tokens, RAM/VRAM y condiciones de caché.

Cada ejecución debe permitir identificar corpus, modelos, configuración y código.
Los experimentos del laboratorio sólo pasan a Consulta después de revisar sus
resultados y limitaciones.

## Orden de trabajo

| Prioridad | Etapa | Entregable principal |
|---|---|---|
| 1 | Laboratorio v1: evaluación y generadores | Dataset revisable, controles con evidencia correcta, trazas y comparador |
| 2 | Reranking y optimización del contexto | Ablaciones de recuperación, segmentación y selección final |
| 3 | RAG multimodal | Muestra de tablas y figuras consultables con evidencia visual |
| 4 | CAG acotado | Comparación RAG/contexto completo/caché KV sobre el mismo subconjunto |
| 5 | KG-RAG y GraphRAG | Grafo pequeño revisado y comparación de preguntas relacionales/globales |
| 6 | Recuperación iterativa | Decisiones de buscar más, pedir aclaración o abstenerse, con límites |

## 1. Laboratorio v1: evaluación y generadores

**Estado:** primera versión implementada en `/lab`: 48 consultas en borrador,
particiones por grupo, tres controles generativos, ablaciones de ranking, snapshots,
comparación A/B y revisión humana. [Uso y límites](laboratory.md).
Pendientes para cerrar esta etapa: revisar referencias, clasificar fallos,
comparar un segundo generador y evaluar una configuración fijada en test.

**Hipótesis:** separar errores de datos, recuperación y generación permite elegir
mejoras con evidencia y evita ajustar el sistema únicamente a ejemplos de la demo.

### Conjunto de evaluación

- Preparar inicialmente 40–60 preguntas, ES/EN, y una muestra de conversaciones.
- Cubrir preguntas directas, categorías numéricas, excepciones, comparaciones,
  tablas, información ausente, versiones y cambios de tema entre turnos.
- Registrar documento, versión, papel, evidencia localizada, respuesta de referencia,
  abstención esperada, tipo de pregunta y estado de revisión.
- Separar desarrollo y test antes de ajustar parámetros. Una pregunta, su traducción
  y sus paráfrasis deben permanecer en la misma partición.
- Los fallos conocidos y casos usados para modificar el sistema pertenecen a desarrollo
  y regresión. Reservar casos nuevos e independientes para test.
- Las preguntas propuestas por un LLM se mantienen como `unreviewed` hasta revisión.
  Distinguir revisión técnica de revisión experta radiológica.

### Controles

Comparar el mismo generador con:

1. Sin evidencia documental, como control de conocimiento previo y abstención.
2. Evidencia correcta seleccionada previamente: control con contexto de referencia.
3. Evidencia recuperada automáticamente por el sistema.

Comparar el modelo actual con otra alternativa compatible con el hardware, fijando
preguntas, contexto, cuantización, opciones y presupuesto de salida. Si falla con
evidencia correcta, registrar el fallo del generador antes de atribuirlo al buscador.
Evaluar por separado la extracción estructurada de campos y la generación libre.

### Mediciones

| Capa | Qué medir |
|---|---|
| Datos | Omisiones, pérdida de relaciones tabulares, figuras pendientes, errores de versión |
| Recuperación | Recall@k, nDCG/MRR, cobertura de evidencia y de sistemas solicitados |
| Generación | Fidelidad por afirmación, contradicciones, condiciones y excepciones conservadas |
| Citas | Existencia del ID, atribución correcta y soporte real de la afirmación |
| Abstención | Casos sin evidencia y abstenciones excesivas en preguntas respondibles |
| Conversación | Resolución de referencias, cambios de tema y arrastre incorrecto de historial |
| Operación | Carga inicial, latencia por fase, tokens, memoria y llamadas al modelo |

Los jueces LLM son ayudas de evaluación, con modelo y prompt registrados; calibrarlos
contra revisión humana y documentar desacuerdos. Informar tamaño de muestra y resultados
por tipo de pregunta, no sólo una media global.

**Entregable:** ejecución reproducible, informe comparativo y primera vista Laboratorio.

**Criterio de avance:** baseline congelado, separación dev/test, fallos clasificados
y resultados que permitan distinguir recuperación insuficiente de generación infiel.
Definir umbrales de aceptación en desarrollo antes de evaluar el test final.

## 2. Reranking y optimización del contexto

**Estado:** primera implementación experimental en `/lab`: Qwen3-Reranker-0.6B
local fijado, pools de hasta 20/40 candidatos, selección trazable y comparación
del cuerpo de contexto padre/pasaje conservando metadatos. Script de comparación
pareada y mediciones de carga, scoring y memoria. [Método y uso](reranking.md).
Pendientes: evaluación con referencias revisadas, ajuste de segmentación/presupuesto
y optimización de la configuración del chat.
El reranker está activado por defecto en el chat, con opción de desactivarlo.
La primera muestra pareada de desarrollo no mejoró el recall anotado; la mejora
sigue sin validar. Resultado y límites en el documento de método.

**Hipótesis:** evaluar pregunta y pasaje conjuntamente mejora la selección final
respecto a la fusión de rangos, con un coste adicional que debe medirse.

- Comparar dense, BM25 y RRF sobre las mismas preguntas y filtros.
- Recuperar 20–40 candidatos y comparar selección final con/sin reranker.
- Evaluar Qwen3-Reranker-0.6B como candidato; fijar revisión y comprobar recursos.
- Cargar modelos por turnos y medir el coste de carga además del forward.
- Mantener k y presupuesto de contexto comparables; estudiar su variación en
  experimentos separados.
- Comparar tamaños de chunk, overlap y agrupación estructural. Los cortes actuales
  de 512 tokens provienen del límite E5, no de un óptimo demostrado para RADS.
- Separar comparación de encoders sobre chunks idénticos de optimización específica
  de cada encoder y su ventana de entrada.
- Evaluar expansión a padres, encabezados/notas y deduplicación de evidencia.
- Estudiar reescritura o traducción de consultas como ablaciones, conservando original,
  negaciones, unidades y términos; no incorporarlas sin medir su aportación.

**Entregable:** comparador visual de candidatos, orden y contexto final.

**Criterio de avance:** mejora documentada por tipo de pregunta y coste; comprobar
que no se pierde evidencia al rerankear ni se perjudica la recuperación ES/EN.

## 3. RAG multimodal

**Estado:** primera muestra integrada en el chat: cuatro regiones PDF con hashes,
visor del recorte original y Qwen3-VL 2B Instruct local. Selección por metadatos,
controles imagen/texto/combinado y trazas separadas. [Uso y método](multimodal.md).
El chat fusiona candidatos textuales y visuales, rerankea sus representaciones
textuales y usa un VLM común con los textos y la imagen original seleccionados.
Pendientes: OCR, referencias visuales revisadas, evaluación más amplia y recuperación
multivector ColPali/ColQwen. No hay embeddings visuales en la implementación actual.

**Hipótesis:** conservar estructura y evidencia visual mejora preguntas que el texto
plano no puede responder correctamente.

- Seleccionar una muestra pequeña: tabla con celdas combinadas y notas, diagrama
  de flujo y figura con etiquetas; preparar preguntas y referencias revisadas.
- Comparar extracción estructurada, OCR y descripción mediante modelo visual.
  La lectura de etiquetas por OCR no demuestra comprensión de flechas o relaciones.
- Conservar imagen/recorte, leyenda, coordenadas cuando existan, documento y versión.
- Identificar las descripciones generadas como contenido derivado, con revisión pendiente.
- Evaluar recuperación visual de páginas o regiones: estudiar ColPali/ColQwen y
  recuperación multivector antes de elegir una implementación.
- Medir memoria, almacenamiento y latencia; comprobar la viabilidad local con una
  muestra antes de ampliar a todas las figuras.
- La interpretación visual requiere un generador con visión o una transcripción
  revisada. El generador textual actual no recibe imágenes.

**Entregable:** visor de tabla/fila o recorte original enlazado desde la respuesta.
Para PDF con coordenadas, resaltar la región; para JATS, mostrar estructura e ID.
No inventar correspondencias de páginas entre XML y PDF.

**Criterio de avance:** evidencia visual localizable, extracción contrastada y mejora
medida frente al baseline textual. La existencia de un embedding visual no valida
la interpretación clínica de una figura.

## 4. CAG acotado y caché de contexto

**Hipótesis:** un subconjunto estable que quepa en contexto puede beneficiarse de
reutilizar estados KV, evitando errores de selección y procesamiento repetido.

- Elegir una sección coherente que quepa con instrucciones, pregunta y salida.
  El texto seleccionado PI-RADS medido ocupa 19.197 tokens del generador y excede
  la ventana actual de 8192; no asumir que cabe el corpus completo.
- Usar el mismo conocimiento permitido y las mismas preguntas para comparar RAG,
  contexto completo y contexto completo con caché KV verificada.
- Fijar generador, opciones y salida. Separar coste de preparación, primera pregunta
  y consultas sucesivas; medir latencia total y tiempo hasta el primer token.
- Verificar hits/reutilización efectiva y consumo RAM/VRAM. Incluir invalidación
  al cambiar documento, instrucciones o versión del modelo.
- Diferenciar caché KV, caché de embeddings y caché de respuestas.
- El chat actual descarga el generador tras responder. El experimento necesita
  una política de residencia y caché explícita, en un entorno aislado del producto.
- Estudiar LMCache y trabajos de reutilización parcial/componible de KV como línea
  de infraestructura; revisar compatibilidad antes de adoptar otro runtime.

**Entregable:** comparador RAG/CAG con calidad, preparación, reutilización y memoria.

**Criterio de avance:** ahorro observado con caché real, calidad contrastada y límites
de contexto documentados. Un prompt completo repetido sin reutilización no demuestra CAG.

## 5. KG-RAG y GraphRAG

**Hipótesis:** relaciones explícitas con procedencia ayudan a preguntas que requieren
conectar evidencia distribuida o sintetizar el conjunto documental.

- Diseñar un grafo pequeño revisado: sistema, edición, modalidad, criterio, condición,
  excepción y evidencia. Preservar negaciones y calificadores.
- Cada arista debe apuntar al documento, versión y fragmento que la respalda.
- Las relaciones propuestas por un LLM requieren revisión; los resúmenes generados
  no se convierten en fuentes normativas.
- Comparar baseline híbrido y KG-RAG sobre preguntas relacionales, con presupuesto
  y conjunto de evidencia comparables. Incluir coste de construcción y actualización.
- Mostrar caminos recuperados y permitir abrir la evidencia de cada relación.
- Después estudiar Microsoft GraphRAG: extracción de entidades/relaciones,
  comunidades, resúmenes y búsquedas local, global y DRIFT.
- Evaluar por separado preguntas concretas, multisección y globales; no asumir que
  una mejora en síntesis global se transfiere a criterios precisos.
- LightRAG y KAG son implementaciones distintas para estudiar cuando haya una
  hipótesis concreta. RadLex puede ayudar a normalizar términos si licencia y
  cobertura lo permiten; no aporta automáticamente las reglas de cada RADS.

**Entregable:** explorador de relaciones con citas y comparación de respuestas.

**Criterio de avance:** beneficio medido en preguntas relacionales/globales, con
errores de extracción y costes explícitos. Usar un grafo no corrige por sí solo
un generador que interpreta mal la evidencia.

## 6. Recuperación iterativa / agentic RAG

**Hipótesis:** decidir cuándo recuperar más o pedir aclaración mejora preguntas
compuestas y conversaciones frente a un único paso de recuperación.

- Definir decisiones: responder, buscar otra sección/sistema, pedir modalidad,
  explicar conflicto de versiones o abstenerse.
- Comenzar con hasta dos iteraciones de recuperación, límites de tiempo y tokens,
  y condiciones de parada observables.
- Mantener el historial conversacional separado de la evidencia y usar sólo citas
  del contexto vigente. Probar referencias como «¿y en el otro sistema?».
- Registrar consultas, herramientas, evidencia añadida y motivo de cada decisión.
- Comparar con recuperación de un solo paso; evaluar el proceso completo, no sólo
  la última respuesta. Medir errores acumulados, coste y latencia.
- Incluir pruebas de instrucciones maliciosas dentro de documentos y de intentos
  de desviar las herramientas fuera de su alcance documental.

**Entregable:** traza legible del proceso de búsqueda y decisiones.

**Criterio de avance:** mejoras sobre el conjunto conversacional revisado sin bucles
ni llamadas innecesarias; abstención y límites de herramientas comprobados.

## Criterios para una demo personal enseñable

- Instalación y preparación reproducibles, con revisiones y hashes fijados.
- Corpus de demostración con condiciones de uso compatibles; datos privados,
  credenciales y originales restringidos permanecen fuera de Git.
- Biblioteca que muestre organismo, sistema, versión, idioma y papel de cada fuente.
- Preguntas de ejemplo y casos de insuficiencia; conservar también fallos conocidos.
- Citas que abran evidencia real, con tabla/fila/figura cuando esté disponible.
- Métodos y límites identificados: extracción estructurada, LLM, caché o grafo.
- Informe de evaluación con configuración, muestra, revisión y límites prácticos.
- Mostrar en Laboratorio por qué funciona una respuesta y qué cambia entre métodos.

## Decisiones aplazadas y control de alcance

- PostgreSQL/pgvector y servicios vectoriales: aplazados; no condicionan los
  experimentos iniciales. Reabrir la decisión al necesitar volumen, filtros,
  actualizaciones o concurrencia que justifiquen la migración.
- Más familias RADS: incorporar con versión, derechos, procesamiento y evaluación
  propios cuando el experimento lo requiera. El manual oficial BI-RADS sigue pendiente.
- Fine-tuning, múltiples frameworks y despliegue público: evaluar cuando exista
  un problema medido y recursos definidos.
- Clasificación de pacientes o recomendaciones clínicas: fuera del alcance actual.

Antes de ampliar esta lista, registrar qué fallo o hipótesis motiva el cambio,
qué entregable produce y cómo se comprobará. Las decisiones técnicas y límites
de hardware se revisan al iniciar cada experimento; los ejemplos de modelos y
frameworks son candidatos, no selecciones definitivas.

## Lecturas vinculadas al plan

Selección de fuentes primarias consultadas para esta ruta; verificar versiones y
disponibilidad del software al implementar cada etapa.

| Tema | Lectura | Utilidad / madurez |
|---|---|---|
| Atribución y fidelidad | [GaRAGe, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.875/) | Evaluación con evidencia de respaldo anotada |
| Conversación | [mtRAG, TACL 2025](https://aclanthology.org/2025.tacl-1.36/) | Preguntas dependientes del historial y evaluación multiturno |
| Diagnóstico modular | [RAGChecker](https://arxiv.org/abs/2408.08067) | Separar recuperación y generación; calibrar sus métricas localmente |
| Reranking | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Candidato multilingüe; comprobar coste y calidad en este corpus |
| Recuperación visual | [ColPali](https://arxiv.org/abs/2407.01449) | Embeddings multivector de imágenes de páginas |
| Recuperación visual eficiente | [HEAVEN, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.54/) | Dos etapas, single-vector y multi-vector; resultados de sus autores |
| CAG | [Artículo original](https://arxiv.org/abs/2412.15605) | Contexto estable y reutilización de estados |
| Infraestructura KV | [LMCache](https://docs.lmcache.ai/) | Implementación; comprobar entorno y compatibilidad |
| Gestión de KV | [Tessera, septiembre de 2026](https://arxiv.org/abs/2609.32999) | Preprint sobre reutilización y gestión de estados en RAG |
| Grafos | [Artículo GraphRAG](https://arxiv.org/abs/2404.16130), [modos de consulta oficiales](https://microsoft.github.io/graphrag/query/overview/) | Grafo, comunidades y búsqueda local/global/DRIFT |
| Vigencia de evidencia | [Lifecycle Faithfulness, GroundLM 2026](https://aclanthology.org/2026.groundlm-1.18/) | Trabajo de workshop sobre evidencia obsoleta, versiones y conflictos |

Documentación del proyecto: [arquitectura](architecture.md), [chat](chat.md),
[chunking](chunking.md), [comparación de enfoques](retrieval_strategies.md),
[fuentes](sources.md) y [bibliografía](bibliography.md).

## Próximo entregable

**Laboratorio v1:** conjunto de evaluación revisable, controles con evidencia correcta,
comparación inicial de generadores e interfaz para inspeccionar recuperación y contexto.
Reranking, multimodalidad, CAG y grafos quedan ordenados detrás de ese baseline.
