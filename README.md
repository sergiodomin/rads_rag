# Radiology Knowledge RAG Lab

RAG local para consultar documentación radiológica mediante un chat con referencias
a fragmentos y páginas del documento. Corpus soportado: **PI-RADS v2.1 (2019)**.
Proyecto experimental, sin validación clínica.

## Arquitectura

`PDF → extracción → chunks → embeddings → búsqueda por similitud → LLM → respuesta con citas`

| Componente | Implementación |
|---|---|
| Backend / interfaz | Python 3.12, HTTP estándar, HTML/CSS/JavaScript |
| Extracción PDF | pypdf, pdfplumber; render con pypdfium2 |
| Segmentación | Ventanas de hasta 512 tokens E5 con overlap del 15 %; alternativa estructural con relación padre-hijo |
| Embeddings | Qwen3 Embedding 0.6B, 1024 dimensiones; alternativa multilingual E5 base, 768 dimensiones |
| Índice y búsqueda | NumPy, vectores normalizados, similitud coseno, top 4 |
| Generación | Qwen3 4B Instruct 2507, Q4_K_M, servido por Ollama local |
| Presupuesto | 8192 tokens de contexto, hasta 512 de salida; conteo con el tokenizer del generador |
| Reproducibilidad | Dependencias en `uv.lock`; revisiones de modelos y hashes en `data/catalog/` |

La recuperación multilingüe admite consultas en español sobre texto inglés sin
traducción intermedia. Encoder y generador se cargan por turnos para limitar memoria.

## Instalación y preparación

Requisitos del flujo actual: **Windows, Python 3.12, uv y GPU NVIDIA con CUDA**.
La preparación descarga los modelos y el runtime portable de Ollama.

```powershell
git clone https://github.com/sergiodomin/rads_rag.git
cd rads_rag
uv sync --locked --extra embeddings
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Incorporar una copia autorizada del PDF. El catálogo verifica su identidad mediante
SHA-256 y número de páginas. Documentos, modelos e índices se preparan localmente
y no se distribuyen en este repositorio.

```powershell
uv run radiology-rag import pirads-2.1-2019 "C:/ruta/PIRADS-2019.pdf" --authorization-note "Copia autorizada para procesamiento local"
uv run python scripts/prepare_tokenizer.py
uv run radiology-rag process
uv run python scripts/prepare_embedding_models.py
uv run --extra embeddings python scripts/build_chat_index.py
uv run python scripts/prepare_local_ollama.py
uv run python scripts/prepare_local_generator.py
```

## Uso

```powershell
uv run --locked --extra embeddings radiology-rag chat --open
```

Interfaz en **http://127.0.0.1:8765**, con selección de encoder y segmentación,
búsqueda documental y respuestas con citas. Configuración mediante `.env` a partir
de [`.env.example`](.env.example); el modo local usa `OLLAMA_NO_CLOUD=1` y no requiere
clave API.

## Desarrollo

```powershell
uv run ruff check src tests scripts
uv run pytest -q
uv run python scripts/check_publish.py --tracked
```

Los [notebooks](notebooks/) cubren exploración del PDF, parsing, chunking y
recuperación bilingüe. Las anotaciones de evaluación se mantienen en
`data/evaluations/`, fuera de Git; plantilla en [`data/templates/`](data/templates/).

## Documentación

- [Chat y configuración](docs/chat.md)
- [Embeddings y recuperación](docs/embeddings.md)
- [Fuentes documentales](docs/sources.md)
- [CAG, GraphRAG y diseño multi-RADS](docs/retrieval_strategies.md)
- [Roadmap](docs/roadmap.md)
