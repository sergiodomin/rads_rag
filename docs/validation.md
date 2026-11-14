# Validación y evidencia de ejecución

Estado actualizado el **2026-11-13**: PI-RADS v2.1 aportado, incorporado y procesado.
Las secciones del 11-11 conservan el historial de ESUR-2012. No son el estado actual
de adquisición del manual. Siguen sin ejecutarse embeddings, retrieval o un LLM.

# Validación de la primera iteración

Fecha: **2026-11-11**. Ejecución local real en Windows con Python **3.12.10** y uv
**0.10.7**. Los resultados siguientes pertenecen al PDF histórico ESUR-2012, no al
manual PI-RADS v2.1. No hay resultados de calidad de RAG en esta iteración.

## Comprobaciones ejecutadas

| Comprobación | Resultado y alcance |
|---|---|
| Instalación | `uv sync` completado y `uv.lock` generado; `uv sync --locked --offline --quiet` completado con el caché local. Extra embeddings no instalado; sin pesos de modelos |
| Lint | `uv run ruff check .`: sin incidencias. El código de notebooks se valida y ejecuta por separado; cachés, datos y resultados excluidos del lint |
| Tests | `uv run pytest -q`: **35 passed**. Configuración, secretos en repr/logs, contrato HTTP, errores y reintentos, permisos, host/ruta, PDF real frente a HTML, hash y cambios de páginas |
| Integridad documental | `radiology-rag verify`: SHA-256 del PDF coincide con el manifiesto y el hash fijado en el catálogo; ID y versión comprobados |
| Notebook 00 | **6 celdas de código**, ejecución completa desde kernel nuevo, sin errores: última medición 1.646 s |
| Notebook 01 | **13 celdas de código**, ejecución completa desde otro kernel nuevo, sin errores: última medición 9.602 s |
| Visualización | Revisadas imágenes completas de páginas físicas 1 y 8, gráfico de perfil y vista de depuración de tabla en página 8; render con PDFium por ausencia de Poppler |
| API pública Ollama | `radiology-rag models`: petición real a `/api/tags` correcta, **23 nombres** devueltos en esa consulta; no supone acceso de la cuenta a cada modelo |
| Falta de clave | `radiology-rag check-config`: `api_key_present=false`. `llm-smoke` termina con error explícito antes de llamar a la API; no se inventó respuesta |
| Exclusiones Git | `git check-ignore` confirma `.env`, PDF, texto procesado, caché y notebook ejecutado excluidos. Repositorio local inicializado; no se ha publicado |

Los tests HTTP emplean **MockTransport explícito**: prueban el comportamiento del
cliente, no la disponibilidad de generación, cuotas o exactitud de respuestas. La
consulta pública de modelos sí fue real. No hubo llamadas de inferencia ni envío de PDF.

El runner usa el intérprete del entorno y kernels temporales independientes, ejecuta
desde `notebooks/` y conserva salidas completas en `output/notebooks/`. Sus hashes de
notebook fuente están en `execution_report.json`. La configuración de IPython/Matplotlib
se mantiene en directorios temporales, sin instalar un kernel global del usuario.

## Fuente incorporada

**Barentsz JO et al., ESUR prostate MR guidelines 2012**, DOI
`10.1007/s00330-011-2377-y`, documento publicado (no manuscrito) según metadatos PMC,
licencia **CC BY-NC**. PDF de **482,998 bytes**, **12 páginas**.

```text
SHA-256: a2e257380f518454278f22da7971d005c9e7b56d66c1a998b48ec163c26e90ff
PDF:     data/raw/esur-2012/guideline.pdf
Origen:  https://pmc-oa-opendata.s3.amazonaws.com/PMC3297750.1/PMC3297750.1.pdf
```

El endpoint del editor devolvió HTML pese al estado HTTP 200; el downloader lo rechazó
y eliminó el archivo parcial. El antiguo servicio PMC `oa.fcgi` devolvió 404: está
retirado tras la migración de agosto de 2026. Se consultó la documentación actual y se
utilizó el depósito oficial PMC Open Access, no un espejo. Véase [fuentes](sources.md).

## Mediciones y observaciones de extracción

- 50,278 caracteres con pypdf; 46,439 con pdfplumber por defecto. Ninguna página bajo
  el umbral exploratorio de 100 caracteres. Esto no demuestra extracción completa.
- Tres imágenes incrustadas detectadas en todo el archivo. Este recuento no cuenta
  necesariamente todas las figuras o gráficos vectoriales.
- El detector de tablas por líneas propone **cero** tablas. La inspección de la página
  física **8**, impresa **753**, muestra una tabla y contenido gráfico/tabular inferior.
  Es un fallo de detección observado, no ausencia de tablas.
- En esa página, pdfplumber por defecto une palabras: **120 tokens de palabra** y
  Jaccard de **0.163** frente a pypdf. Con `x_tolerance=1` o ratio `0.1`, **320 tokens**
  y Jaccard **1.0** en esta comparación. No se ha medido exactitud semántica ni se
  afirma una mejora general del parser. Jaccard ignora orden y repeticiones.
- El experimento de tablas en la página 8 produce: líneas, 0 candidatos; texto por
  defecto, 1 candidato de 48 filas/1 columna; texto con tolerancia 1, 1 candidato de
  48 filas/9 columnas. Al contrastar el render, esas estructuras no reconstruyen
  correctamente la tabla. Más columnas no implica mejor extracción.
- El elemento tabular dentro de la imagen inferior de esa página no aparece como
  texto recuperado por pypdf. Se ha comprobado en el render y la extracción; aún no
  hay OCR ni extracción multimodal.
- La página física 1 corresponde a la impresa 746 y contiene prosa a dos columnas y
  afiliaciones. La revisión del orden completo de lectura de todas las páginas queda
  pendiente; no se ha validado una jerarquía de secciones.

Estos resultados se guardan en `data/processed/esur-2012/`: perfil CSV, comparaciones
de tolerancia y tablas, `exploration.json`, páginas JSONL y PNGs. Los registros por
página preservan fuente, versión y hash; las secciones permanecen en `null`.
La revisión visual realizada por el agente **no sustituye una revisión humana** ni
constituye un gold standard de extracción o de respuestas médicas.

## No probado y límites

No se ha probado inferencia real en Ollama Cloud por ausencia de clave/modelo configurado;
tampoco embeddings, retrieval, reranking, calidad de respuestas, clasificación o grafos.
El manual ACR v2.1 y sus suplementos siguen pendientes de autorización y no se descargaron.
No se eludió autenticación. No hay benchmark ni métricas de precisión clínica.

Sólo se ha verificado este entorno Windows/Python 3.12.10. El lock declara un rango
de Python 3.12–3.13, pero no se ejecutó una matriz de sistemas/versiones. Los tiempos
incluyen arranque local; no son un benchmark estable. El sandbox bloqueó inicialmente
descargas y comunicación del kernel: se repitieron con los permisos del entorno.

Siguiente acción: notebook **02**, con 6–10 regiones revisadas manualmente y comparación
de parsing/chunking manteniendo condiciones, notas y relaciones de tabla. La página 8
es un caso observado que permite medir pérdidas concretas.

## Iteración 2 · Parsing y chunking (2026-11-11)

Implementado y ejecutado `02_parsing_and_chunking.ipynb`: **16 celdas de código**, sin
errores, última ejecución **51.509 s**. `ruff check .` correcto y **48 tests passed**.
La suite ahora comprueba también ventanas por tokens, Unicode/offsets, cobertura,
overlap, versiones de IDs, secciones, tablas multipágina, padres largos, anclas y
cortes forzados. Uno de los tests utiliza el tokenizer real local cuando está preparado;
el resto no necesita descargas. Ninguna prueba requiere una API key.

Se añadió Tokenizers **0.23.3**, con resolución fijada en `uv.lock`, y se descargó sólo
`tokenizer.json` de E5 (**17,082,660 bytes**), commit y hash en
`data/catalog/tokenizers.v1.json`. No se descargaron ni ejecutaron pesos del encoder.
La sección semántica quedó desactivada y se registra `executed=false`, `llm_calls=0`.
La prueba de algoritmo con cortes forzados no se presenta como inferencia semántica.

El mapa específico del documento se contrastó visualmente, incluyendo páginas 4–6:
se separaron rótulos del diagrama vectorial y pies de figuras de la prosa del corpus.
Hay **4 objetos visuales pendientes** (3 imágenes y 1 diagrama vectorial). Todos se
registran; no se inventaron transcripciones ni relaciones. La revisión humana del
mapa, secciones, tablas y microreferencias sigue pendiente.

Referencia de ejecución final:

```text
run_id: 65c27174feb16711
data/processed/esur-2012/02_chunking/runs/65c27174feb16711/
```

Se obtuvieron **93 bloques**, de los que **80** entran en el corpus seleccionado:
**29,209 caracteres**. Referencias, avisos y pies de figura quedan conservados fuera
de la comparación. La cobertura siguiente se refiere a ese texto, no al PDF entero.

| Configuración | Chunks | Factor de copias | Grupos de tabla íntegros | Chunks >512 |
|---|---:|---:|---:|---:|
| Fijo 256, sin overlap | 31 | 1.000 | 2/4 | 0 |
| Fijo 256, overlap 15% | 37 | 1.175 | 2/4 | 0 |
| Fijo 512, sin overlap | 16 | 1.000 | 2/4 | 0 |
| Fijo 512, overlap 15% | 18 | 1.167 | 1/4 | 0 |
| Fijo 768, sin overlap | 11 | 1.000 | 2/4 | 10 |
| Fijo 768, overlap 15% | 12 | 1.167 | 2/4 | 12 |
| Estructural 256 | 39 | 1.000 | 4/4 | 2 |
| Estructural 512 | 26 | 1.000 | 4/4 | 2 |
| Estructural 768 | 20 | 1.000 | 4/4 | 7 |

Las nueve configuraciones cubren el 100% del texto seleccionado. El estructural 512
conserva los cuatro grupos de tabla como texto, pero dos padres miden **746 y 998
tokens** y no caben completos en E5. Tampoco se ha validado semántica de filas y
columnas. Esto impide atribuir a esa métrica una mejora de retrieval. En este corpus,
el overlap 15% tampoco aumenta monotónicamente las tablas completas: al desplazar
límites puede mejorar unos casos y empeorar otros.

Las diez microregiones, transcritas del render y pendientes de revisión humana, dan
**5/10** coincidencias normalizadas exactas con pdfplumber por defecto y **9/10** con
tolerancia 1. La restante contiene texto dentro de una imagen y sigue sin extraerse.
Son casos pequeños elegidos conociendo fallos previos; no es un benchmark independiente.
Se usó `within_bbox` para excluir caracteres parcialmente fuera del crop de referencia.

Se ejecutaron cuatro micropruebas de contención de anclas, separando presencia en
cualquier chunk y presencia en uno dentro de la ventana. El caso de tabla multipágina
muestra la diferencia entre conservar un padre y poder vectorizarlo completo. No hay
Recall@k, evaluación de QA ni inferencias sobre exactitud clínica.

Los gráficos y diez crops se revisaron visualmente. Los JSONL verifican texto contra
offsets del corpus, IDs únicos, fuente/hash y presupuesto. El manifiesto registra
huellas de documento, mapa, referencias, tokenizer, notebook, módulos y lockfile;
`latest.json` señala la última ejecución. Git excluye tokenizer, PDFs y derivados.

La próxima fase es **03: embeddings y recuperación**. La comparación semántica de 02
sigue siendo opcional y sin ejecutar; los padres largos requieren tratamiento explícito
antes de utilizarlos con E5. No se han modificado `.env` ni las fuentes ACR pendientes.


## Iteración 3 · Manual PI-RADS v2.1 (2026-11-13)

Archivo aportado: `PIRADS-2019.pdf`, **76 páginas**, **3,019,107 bytes**.
El usuario confirmó la cobertura para incorporarlo y procesarlo. Se mantuvo el original
y se creó una copia inmutable más manifiesto en `data/raw/pirads-2.1-2019/`.
SHA-256: `594e618213e76205a5a91ca3df488274b0f6f42325b173581577da556cf2b4f2`.
No se efectuó una descarga del manual ni se envió su contenido a Cloud.

Se inspeccionaron renders de las 76 páginas y ampliaciones de las regiones de referencia.
Mapa específico ligado al hash, con numeración física y editorial: i–vii y 1–69.
La revisión humana de mapa, transcripciones y relaciones tabulares permanece pendiente.

Ejecución baseline: **`a7bd824ca0aec05d`**.
Ruta: `data/processed/pirads-2.1-2019/02_chunking/runs/a7bd824ca0aec05d`. `latest.json` apunta a este resultado completo.

- **534 bloques totales**, **224 seleccionados**,
  **85,110 caracteres** para los experimentos.
- Se conservan páginas brutas y bloques excluidos (bibliografía, créditos, ejemplos,
  avisos y pies). No se presenta la selección como cobertura íntegra del manual.
- **54 imágenes informativas pendientes**, sin OCR/transcripción;
  **84 decorativas registradas aparte**.
- Nueve colecciones reales, cobertura 100% del texto seleccionado, offsets exactos e IDs únicos.
- 512 tokens con overlap 15%: **50 chunks**, factor de copias **1.177**; ninguno supera 512.
- Estructural 512: **55 chunks**, **14/14** grupos de tabla íntegros; un padre de
  **724 tokens** excede E5. No se trunca ni se declara listo para el encoder.
- Microreferencias: 9/10 textos exactos tras normalizar con ambas configuraciones;
  el control de imagen no se extrae. Son diagnósticos pequeños, no precisión del manual.
- `radiology-rag verify` verifica PI-RADS y ESUR. Repetir `process` reutiliza la misma
  huella tras comprobar todos los hashes de artefactos.
- Regresión ESUR: `process esur-2012` conserva 80 bloques seleccionados y 29,209 caracteres.

Comprobaciones de código: **58 tests passed**, `ruff check .` sin errores.
Se añadieron pruebas de origen local, idempotencia, identidad distinta, hash alterado,
bloqueos, fallo de copia, capitulares y anclas delimitadas por página.

Notebooks ejecutados sobre PI-RADS (fuentes sin salidas; copias en `output/notebooks/`):

| Notebook | Celdas de código | Segundos |
|---|---:|---:|
| 00_project_overview.ipynb | 6 | 21.925 |
| 01_document_exploration.ipynb | 13 | 54.024 |
| 02_parsing_and_chunking.ipynb | 16 | 124.386 |

Los informes `execution_report_pirads_first.json` y `execution_report.json` conservan
las huellas ejecutadas. 01 se repitió después de añadir la etiqueta impresa también
a su exportación de páginas. 02 contrastó los IDs de sus algoritmos visibles con
las nueve colecciones publicadas por la misma función que usa la CLI.

No hay encoder ejecutado, embeddings persistentes, índice vectorial, recuperación
evaluada ni respuestas de LLM. Siguiente fase: 03, con tratamiento explícito de padres
largos y selección de evidencia revisada. Los otros documentos ACR siguen pendientes.

## Iteración 4 · Embeddings y recuperación (2026-11-14)

Después de reparar Windows con DISM y SFC y del reinicio manual del usuario,
se comprobó el arranque a las 18:35:05 Europe/Madrid. Ambas GPU tenían estado OK,
Intel 31.0.101.2145 y NVIDIA 32.0.15.9186; las marcas CBS y Windows Update de
reinicio pendiente estaban desactivadas. La prueba mínima de CUDA pasó en la
RTX 2060, capability 7.5, PyTorch 2.14.1+cu126 y runtime CUDA 12.6.

La primera importación de PyTorch fue lenta. Una captura local del proceso de
prueba mostró espera en LoadLibraryExW/NtCreateSection durante la carga de DLL.
No identifica su causa. El proceso terminó y verificó el producto de matrices
64x64 float16 en cuda:0; no se desactivaron controles de seguridad.

Notebook `03_embeddings_and_retrieval.ipynb`: **13 celdas de código, PASS,
61,403 s**. Informe `output/notebooks/execution_report_03_first.json` y copia
ejecutada en ese directorio. SHA-256 fuente:
`d96a83f2a5dee91f0addc241eaae0b9a66a754cb096363e35966c6f1222c73dd`.

Ejecución publicada: **ae487cf495cff38a**, sobre chunking **a7bd824ca0aec05d**.
Piloto fijado antes de calcular rankings: 12 casos / 24 consultas, 10 casos
respondibles (6 dev, 4 test) y 2 controles; traducciones ES/EN en la misma partición.
Anotaciones y suficiencia de evidencia con revisión humana pendiente.

- 50 pasajes fijos y 56 estructurales, tras crear dos hijos del padre de 724 tokens.
- 106 pasajes únicos por encoder; 24 consultas con formato y 24 sin formato.
- E5: dimensión 768; Qwen3: 1024; forward/pooling reales, CUDA, float16, batch 1.
- Sin truncación; máximo observado E5 512 tokens y Qwen3 507 en los pasajes.
- E5 codificó los pasajes en 1,828 s; Qwen3 en 8,964 s, sin contar carga y publicación.
- Máximo de VRAM asignada por PyTorch: 555,9 y 1217,0 MiB respectivamente.
- Máximo de RAM del proceso: aproximadamente 3,05 GiB.
- Cuatro matrices de colección publicadas y seis matrices originales en caché.
- Verificados todos los hashes de índice e informe, dimensiones, filas, finitud y normas.
- Algoritmos visibles de pooling, búsqueda y métricas contrastados mediante asserts.
- **65 tests passed**, Ruff sin errores. Nuevas pruebas: coseno y empates,
  vectores inválidos, evidencia única frente a overlap, anclas por página y
  prohibición de calibrar con test.

Los resultados completos y su interpretación están en [embeddings.md](embeddings.md).
El control visual de test en inglés supera incorrectamente el umbral en ambas
colecciones estructurales. Se conserva el fallo; no se reajustó el umbral con test.
No hay generación, llamadas a LLM ni envío de documentación a Cloud en 03.
La ejecución completa terminó sin reinicio; eso no acredita estabilidad prolongada
ni confirma la causa de los tres fallos anteriores del watchdog.

## Interfaz local de chat (2026-11-14)

Se añadió `radiology-rag chat`, con servidor local, frontend estático, búsqueda
exacta sobre los índices de 03 y generación mediante el cliente Ollama Cloud.
84 pruebas superadas, Ruff sin errores y `node --check` correcto. Las pruebas
añadidas cubren fuentes transmitidas al generador simulado, citas inexistentes,
historial sin roles privilegiados, límite de pregunta sin truncación, exclusión
mutua, ausencia de llamadas Cloud en modo búsqueda, HTTP 402 sin reintentos,
acceso HTTP local, origen, rutas privadas, ciclo de vida de staging y ocultación
de claves en el diagnóstico manual. Se corrigió un cierre TCP intermitente al
rechazar peticiones POST: el cuerpo acotado se consume antes de responder al error.

Se detectó y corrigió un problema real de permisos: TemporaryDirectory creaba
directorios con una DACL privada que persistía al publicar por rename. El proceso
fuera del sandbox no podía leer los manifiestos. Se restableció la herencia de
los artefactos afectados y se implementó artifact_stage para futuros resultados,
incluida la publicación de 03. Los bytes y hashes del índice histórico se conservan;
la fuente actual de 03 contiene ese cambio de publicación y no se ha vuelto a ejecutar.

La búsqueda real de selección de lesión índice devuelve la página 19 del PDF.
Los rankings observados de esa consulta coinciden con los guardados por 03.
La pregunta más general sobre información clínica no recuperó el pasaje esperado
en top 4: se registra el fallo y no se presenta como cobertura garantizada.

Dos peticiones reales a Ollama Cloud devolvieron HTTP 402. La generación end-to-end
está pendiente de resolver ese rechazo del proveedor; listar modelos sí funciona.
No se declara una respuesta Cloud validada ni se ha repetido la generación sin
aviso del usuario. `scripts/check_ollama.py` permite diagnosticarlo manualmente,
con una sola petición mínima y sin documentos ni historial.

El navegador de esta sesión no está accesible; no se han inspeccionado visualmente
la interfaz ni su disposición móvil. Comprobaciones HTTP en
`output/chat_validation.json`. Véase [la guía del chat](chat.md).

## Generador local en RTX (2026-11-15)

El usuario pidió usar un generador que cupiera en su RTX 2060 de 6 GiB. Se preparó
Ollama portable 0.40.16 desde el ZIP oficial, SHA-256
`50fcf5ad54277f2fa5d173ebfc93bcb017adbc0c9ef2314daabc96ca9e6aa64e`.
No se instalaron servicios ni se cambiaron drivers. Se incluyó CUDA 12 y se
omitieron los backends CUDA 13, Vulkan, ROCm y MLX de esa copia portable.
El registro confirmó CUDA, compute 7.5 y NVIDIA GeForce RTX 2060.

Modelo final: **qwen3:4b-instruct-2507-q4_K_M**, digest
`0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`.
Se eligió la variante Instruct después de que el tag híbrido y una prueba de
plantilla continuaran generando razonamiento al pedir respuesta directa.
No se usa esa variante híbrida en el chat. El modelo final respondió exactamente
OK en la prueba mínima y terminó normalmente.

Con contexto 8192, `/api/ps` informó size y size_vram idénticos:
**3.721.563.996 bytes**, 100 % GPU. No se infiere esa colocación sólo por detectar CUDA.
El tokenizer del generador está fijado en Qwen/Qwen3-4B-Instruct-2507, revisión
`cdbee75f17c01a7cc42f958dc650907174af0554`, con archivos y SHA-256 en el catálogo.

La prueba completa de `scripts/check_local_rag.py` pasó:

- Pregunta documental española de selección de lesión índice; cuatro fragmentos.
- Respuesta española con referencia localizable [S2] y finalización stop.
- 35,657 s totales, incluyendo carga inicial del encoder; generación 8,135 s.
- 2327 tokens de entrada reales, presupuesto conservador 2455, ventana 8192.
- 99 tokens de salida, máximo configurado 512; no se recortó salida ni contexto.
- Cero llamadas Cloud. Al terminar, encoder liberado y `/api/ps` sin modelos.
- Memoria disponible posterior de unos 4 GiB y VRAM posterior de unos 290 MiB.

**89 pruebas superadas**, Ruff y sintaxis JS correctos. Nuevas pruebas de HTTP
loopback sin clave, bloqueo explícito de Cloud, no transmisión de claves antiguas
a localhost, exclusión de entradas remotas del servidor local, presupuesto de
contexto sin truncación y liberación del encoder antes de generar.

Evidencia: `output/chat_validation_local.json`, `output/chat_local_example.json`,
`output/diagnostics/local_generator_smoke.json`. La existencia de IDs de cita
no demuestra por sí sola fidelidad clínica. No se amplió el piloto ni se ejecutó
un benchmark independiente de calidad del generador. El notebook 04 sigue pendiente.

## Preparación de la publicación (2026-11-15)

Ruff y **92 pruebas** superadas. Se añadieron controles del auditor de publicación
y una prueba de procesamiento con PDF sintético sin referencias privadas. No se
repitió inferencia en GPU para publicar documentación y código.

Las microreferencias y el piloto con anclas documentales permanecen locales,
ignorados por Git junto a PDF, texto extraído, índices, modelos y diagnósticos.
El procesamiento sin referencias declara que no hay evaluación de referencia.
`scripts/build_chat_index.py` permite preparar índices sin ese piloto y marca
`evaluation_executed: false`; no sustituye la ejecución experimental de 03.
Este constructor nuevo se comprobó estáticamente, sin reconstruir el índice activo.

`scripts/check_publish.py --staged` revisa los blobs exactos del índice Git,
incluidos tamaños, patrones de credenciales, coincidencias con claves configuradas
y ausencia de salidas de notebooks. El manifiesto de auditoría permanece en
`output/`; no contiene valores secretos. El escáner no garantiza ausencia de toda
clase de dato sensible y se complementa con revisión de los archivos publicables.

La primera ejecución CI detectó que `pytest` como ejecutable no encontraba el
directorio de scripts, aunque `python -m pytest` sí lo hacía. Se añadió la raíz
del proyecto a `pythonpath` en la configuración de pytest, de modo que ambas
formas de invocación puedan importar los diagnósticos probados.
