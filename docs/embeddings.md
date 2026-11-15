# Embeddings y recuperación: paso 03

Ejecución real del 14-11-2026: `ae487cf495cff38a`, sobre la ejecución de chunking
`a7bd824ca0aec05d` de PI-RADS v2.1. Notebook completo: 13 celdas de código,
61,403 segundos. La fuente permanece sin salidas; la copia ejecutada está en
`output/notebooks/03_embeddings_and_retrieval.ipynb`.

## Datos y tratamiento de longitud

Se reutiliza el corpus seleccionado de 02: 224 bloques y 85.110 caracteres.
Las 54 imágenes informativas siguen sin OCR ni transcripción. Este paso recupera
texto del manual aportado, con hash y cobertura de uso ya confirmada.

Se comparan dos colecciones sobre los mismos textos:

- Fijo: 512 tokens E5, overlap 15 %, 50 pasajes.
- Estructural: 55 padres; el de 724 tokens se divide en dos hijos trazables,
  resultando 56 pasajes. Los padres originales de 02 se conservan.

Los cortes usan offsets del texto original. Se comprueba cobertura completa del
padre y longitud con prefijo `passage: ` y tokens especiales; no hay truncación
silenciosa. Un corte mecánico puede separar relaciones de una tabla aunque
conserve todos sus caracteres. Las dos colecciones suman 106 pasajes únicos.

## Tokenizers, encoders y similitud

El tokenizer E5 fijado por revisión y SHA-256 sigue definiendo los cortes. Para
inferencia, cada encoder usa el tokenizer incluido en su propia revisión local.
Tokenizar convierte texto en IDs; el transformer usa sus pesos para producir
estados ocultos. No se ha entrenado ni ajustado ninguno de los modelos.

| Encoder | Revisión | Vector | Pooling | Formato de consulta |
|---|---|---:|---|---|
| intfloat/multilingual-e5-base | d128750597153bb5987e10b1c3493a34e5a4502a | 768 | Media con máscara | `query: ` |
| Qwen/Qwen3-Embedding-0.6B | 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 | 1024 | Último token válido | Instrucción y `Query:` fijados en catálogo |

E5 es de arquitectura XLM-RoBERTa y Qwen3 utiliza su propio transformer. El
catálogo `data/catalog/embedding_models.v1.json` fija revisiones, inventario y
hashes de pesos. Se ejecutan en `cuda:0`, float16, batch 1 y dos hilos CPU,
con un modelo cargado cada vez. Pooling y normalización se calculan en float32.
Los vectores finales se guardan como NumPy float32 normalizados a norma L2 igual a 1.

La similitud coseno se calcula mediante producto matricial de vectores normalizados.
La búsqueda es exacta y conserva el orden de los pasajes en los empates. No hace
falta un servicio de base de datos para 50–56 filas por colección. Se mantienen
filas, IDs, páginas físicas, etiquetas impresas, secciones, offsets y padres.

Se utilizan PyTorch 2.14.1+cu126, CUDA 12.6, Transformers y NumPy. El lockfile
conserva las versiones completas. Sentence Transformers no se importa: durante la
preparación Windows bloqueó una DLL de SciPy en esa ruta; la implementación directa
permite inspeccionar y contrastar el forward y el pooling de ambos encoders.

## Costes medidos en la RTX 2060 de 6 GiB

| Encoder | Codificar 106 pasajes | Codificar 24 consultas con formato | Máximo de VRAM asignada en pasajes |
|---|---:|---:|---:|
| E5 | 1,828 s | 0,416 s | 555,9 MiB |
| Qwen3 | 8,964 s | 2,197 s | 1217,0 MiB |

Son tiempos de inferencia local, con sincronización CUDA, separados de carga,
hashes, tokenización previa, publicación y arranque de Python. El máximo de RAM
del proceso alcanzó aproximadamente 3,05 GiB. Esa lectura no representa la RAM
de todo Windows. La VRAM es el máximo asignado por PyTorch por codificación;
no incluye todos los recursos que puede contabilizar nvidia-smi.

También se codificaron las 24 consultas sin formato como ablación. La caché incluye
texto y orden de filas, modelo, revisión, hashes, versiones, código, prefijo,
dispositivo, precisión y batch. Al reutilizarla se validan hash, forma y normas.
Sus tiempos pertenecen a la creación original, no a la lectura de caché.

## Piloto y resultados

Hay 12 casos, cada uno traducido a español e inglés: 24 consultas. Diez casos son
respondibles con anclas textuales: seis de desarrollo y cuatro de test. Los otros
dos son controles: fuera de dominio en desarrollo e información visual ausente
en test. Las traducciones comparten partición; no son observaciones independientes.
El piloto se escribió antes de calcular rankings y tiene revisión humana pendiente.

La relevancia exige contener una ancla completa localizada por página. Recall
cuenta pasajes relevantes; EvidenceRecall cuenta unidades únicas de evidencia y
evita que el overlap multiplique la misma ancla. MRR mide el rango recíproco del
primer pasaje relevante. Estas métricas evalúan recuperación de anclas y no
garantizan que estén todas las condiciones necesarias para una respuesta médica.

Resultados de test a k=5, con formato de consulta oficial del experimento:
**cuatro preguntas respondibles por idioma y configuración**.

| Encoder | Colección | Idioma | Recall@5 | MRR@5 | EvidenceRecall@5 |
|---|---|---|---:|---:|---:|
| E5 | Fijo | EN | 0,750 | 0,583 | 1,000 |
| E5 | Fijo | ES | 0,500 | 0,375 | 0,500 |
| E5 | Estructural | EN | 1,000 | 0,396 | 1,000 |
| E5 | Estructural | ES | 0,500 | 0,175 | 0,500 |
| Qwen3 | Fijo | EN | 1,000 | 0,875 | 1,000 |
| Qwen3 | Fijo | ES | 0,875 | 0,583 | 1,000 |
| Qwen3 | Estructural | EN | 1,000 | 0,688 | 1,000 |
| Qwen3 | Estructural | ES | 0,500 | 0,500 | 0,500 |

En este piloto Qwen3 con cortes fijos recupera todas las anclas de test en ambos
idiomas entre los cinco primeros. El estructural pierde evidencia para algunas
consultas en español. E5 también presenta diferencias ES/EN. El tamaño y la
selección deliberada de preguntas impiden establecer una clasificación general
de modelos o elegir parámetros ajustándolos al test.

El umbral de abstención se ajustó sólo en desarrollo, por balanced accuracy.
Ambos modelos con colección estructural aceptan incorrectamente la consulta
visual de test en inglés, aunque el corpus textual no contenga el dato solicitado.
Un score alto no verifica suficiencia documental. Antes del baseline RAG conviene
ampliar controles y revisar la evidencia; no se han reajustado umbrales con test.

## Artefactos y reproducción

El puntero canónico es `data/runs/pirads-2.1-2019/03_embeddings/latest.json`.
Referencia tanto el informe completo como el índice correspondiente:

- `data/indices/pirads-2.1-2019/03_dense/runs/ae487cf495cff38a/`: cuatro matrices
  de pasajes, sus JSONL y matrices de consultas por modelo.
- `data/runs/pirads-2.1-2019/03_embeddings/runs/ae487cf495cff38a/`: métricas,
  rankings, costes, scores, calibración, evidencia, figura y manifiestos.
- `data/indices/embedding_cache/`: matrices originales, incluidas las consultas
  sin formato, con manifiesto por identidad de entrada.

La publicación usa staging, bloqueo y reemplazo del puntero al completar ambos
destinos. Se verificaron todos los hashes exportados, dimensiones, número de filas,
finitud y normalización. Los derivados locales y PDFs siguen excluidos de Git.

En VSCode puede abrirse la copia ejecutada para revisar resultados sin repetir
inferencia. Para modificar el experimento, abrir la fuente en `notebooks/`,
seleccionar `.venv/Scripts/python.exe` y ejecutar de arriba abajo.

```powershell
uv sync --locked --extra embeddings
uv run python scripts/prepare_embedding_models.py
uv run --extra embeddings python scripts/execute_notebooks.py 03_embeddings_and_retrieval.ipynb --timeout 1800
```

El notebook requiere los modelos preparados localmente y no realiza llamadas LLM
ni descargas durante su ejecución. Se verificó una ejecución completa con CUDA.
