# Bibliografía y selección tecnológica

Selección inicial del **2026-11-11**, actualizada el **2026-11-15**. Se distinguen manuales, artículos originales y
documentación de software. Los resultados de sus autores no son resultados del laboratorio.

## Radiología

1. **ACR / ESUR / AdMeTech. PI-RADS v2.1 (2019)**. [Portal oficial y archivo de versiones](https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/PI-RADS).
   Fuente normativa de la fase actual cuando se autorice su uso. El catálogo incluye
   el PDF actual y v2 de 2015. [Condiciones de ACR](https://www.acr.org/Legal).
2. **Turkbey B et al.** Prostate Imaging Reporting and Data System Version 2.1: 2019
   Update of Prostate Imaging Reporting and Data System Version 2. *European Urology*
   76(3), 340–351 (2019). [DOI: 10.1016/j.eururo.2019.02.033](https://doi.org/10.1016/j.eururo.2019.02.033).
   La página ACR enlaza la publicación original; el editor devolvió 403 en esta sesión.
   Referencia localizada; no se descargó texto completo ni se presume licencia abierta.
3. **Barentsz JO et al.** ESUR prostate MR guidelines 2012. *European Radiology* 22,
   746–757 (2012). [Editor](https://link.springer.com/article/10.1007/s00330-011-2377-y),
   [licencia en PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3297750/).
   Fuente primaria histórica disponible para el laboratorio no comercial. No usar para
   atribuir criterios actuales a v2.1. Es el corpus inicial de exploración.

## RAG y modelos

4. **Lewis P et al.** Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
   (2020). [Original](https://arxiv.org/abs/2005.11401). Referencia fundacional, no guía
   suficiente para seleccionar todo el stack actual.
5. **Wang L et al.** Multilingual E5 Text Embeddings: A Technical Report (2024).
   [Original](https://arxiv.org/abs/2402.05672),
   [modelo del autor](https://huggingface.co/intfloat/multilingual-e5-base).
   Candidato inicial local: 768 dimensiones, límite de 512 tokens, prefijos `query:` y
   `passage:` incluso con consultas en español; normalizar antes de producto escalar.
   Licencia MIT según la ficha del modelo. Comparación local ejecutada en 03;
   resultados y límites en [embeddings](embeddings.md).
6. **Qwen Team.** Qwen3 Embedding: Advancing Text Embedding and Reranking Through
   Foundation Models (2025). [Original](https://arxiv.org/abs/2506.05176),
   [ficha oficial 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
   Comparador más reciente: 0.6B parámetros, contexto 32k, hasta 1024 dimensiones,
   multilingüe y Apache-2.0. Más pesado que E5 base; seguir instrucciones de consultas
   y registrar truncación, RAM y latencia. Pesos locales fijados y comparación
   ejecutada en 03; encoder predeterminado del chat.

## Software y coste de adopción

| Componente | Decisión / alternativa | Razón y límite | Fuente primaria |
|---|---|---|---|
| Entorno | uv y Python 3.12 | Lockfile y entorno local; verificar wheels al actualizar | [uv](https://docs.astral.sh/uv/guides/projects/) |
| PDF | pypdf + pdfplumber | Inventario y geometría/tablas; no reconstruyen semántica de forma fiable por sí solos | [pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html), [pdfplumber](https://github.com/jsvine/pdfplumber) |
| Render | pypdfium2 | Evita instalación de Poppler en Windows; imagen necesaria para revisar extracción | [proyecto oficial](https://github.com/pypdfium2-team/pypdfium2) |
| LLM | Qwen3 4B Instruct con Ollama local y httpx | Cuantización Q4_K_M, contexto 8192; modelo descargado aparte | [Modelo](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [chat](https://docs.ollama.com/api/chat) |
| Embeddings | Sentence Transformers, extra opcional | Local; descarga de pesos y RAM/CPU son costes reales. Fijar revisión en 03 | [documentación mantenida](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) |
| Orquestación | Python directo ahora | Facilita ver los algoritmos; reevaluar LangChain/LlamaIndex sólo ante una necesidad concreta | Decisión del laboratorio |
| Índice | Matrices NumPy, búsqueda exacta | Corpus pequeño; no requiere servicio ni índice aproximado | Decisión del laboratorio |

El generador y su tokenizer se fijan en `data/catalog/local_generator.v1.json`.
El modo local no requiere clave. El acceso a una etiqueta Cloud no acredita cuota
de generación; el modo remoto es opcional y no forma parte del arranque predeterminado.

## Lecturas para fases posteriores, sin implementación aún

- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/): documentación oficial del
  pipeline de indexación y recuperación con comunidades.
- [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG): repositorio de la implementación
  específica, con artículo enlazado y modos de recuperación.
- [OpenSPG/KAG](https://github.com/OpenSPG/KAG): implementación de KAG sobre OpenSPG,
  con razonamiento y esquema. No tratar como sinónimo de GraphRAG.
- [RadLex, RSNA](https://www.rsna.org/practice-tools/data-tools-and-standards/radlex-radiology-lexicon):
  terminología radiológica y licencia específica, enlazada por RSNA. Candidato de
  normalización, pendiente de evaluar cobertura y correspondencias con criterios.
- [PMC Cloud Service](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/): acceso automatizado
  oficial al PDF abierto empleado. Sustituye los antiguos endpoints retirados en 2026.

La bibliografía avanzada se ampliará antes de cada etapa (BM25/RRF, rerankers,
evaluación, contextual retrieval y RadLex). Esta lista inicial no se presenta como una
revisión sistemática ni como validación comparativa de frameworks.

## CAG y GraphRAG: ampliación del 2026-11-15

- **Chan BJ et al.** Don't Do RAG: When Cache-Augmented Generation is All You Need
  for Knowledge Tasks. [Artículo original](https://arxiv.org/abs/2412.15605).
  Contexto estable y caché de estados; comprobar capacidad y reutilización efectiva.
- **Edge D et al.** From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization. [Artículo original](https://arxiv.org/abs/2404.16130).
  Grafo y resúmenes de comunidades; sus resultados no son una evaluación de PI-RADS.

[Diseño experimental y ampliación multi-RADS](retrieval_strategies.md).

## Incorporado en 02 (2026-11-11)

- [Tokenizers, repositorio oficial](https://github.com/huggingface/tokenizers): se usa
  la versión estable 0.23.3 fijada por `uv.lock`, no la API 1 prerelease. Contar tokens
  no ejecuta un encoder ni produce embeddings.
- [E5, ficha del autor](https://huggingface.co/intfloat/multilingual-e5-base): tokenizer
  fijado al commit `d128750597153bb5987e10b1c3493a34e5a4502a`; prefijo de pasaje y
  especiales incluidos en el presupuesto de 512. La prueba de 768 registra incompatibilidad.
- [SentenceTransformer, API oficial](https://sbert.net/docs/package_reference/sentence_transformer/model.html):
  parámetros de revisión, cache y carga sólo local comprobados documentalmente para
  la celda semántica opcional. No se ha probado inferencia del encoder en esta iteración.
