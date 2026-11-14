# Chat documental local

La interfaz reutiliza el índice completo del paso 03. El servidor escucha sólo
en `127.0.0.1:8765`; no se ha publicado el proyecto en Internet. Los archivos
HTML, CSS y JavaScript están en `src/radiology_rag/web/static/`. El backend usa
la biblioteca estándar HTTP de Python, sin instalar otro framework web.

## Abrir y diagnosticar

Estado actual, 15-11-2026: **RAG completamente local probado**, con generador
`qwen3:4b-instruct-2507-q4_K_M` y runtime portable Ollama 0.40.16. La prueba
respondió en español con referencia [S2] al texto recuperado. No hay coste de API
ni envío de pregunta, historial o fragmentos a Cloud en este modo.

Desde la raíz del proyecto:

```powershell
uv run --locked --extra embeddings radiology-rag chat --open
```

Cerrar con Ctrl+C en esa terminal. Puede omitirse `--open` y abrir manualmente
`http://127.0.0.1:8765`; `--port` permite cambiar el puerto. Requiere los modelos
y el índice preparado en 03. En esta copia ya existen. El comando también arranca
el runtime portable si es necesario; no instala servicios de Windows.

Preparación para una copia nueva en Windows:

```powershell
uv run python scripts/prepare_local_ollama.py
uv run python scripts/prepare_local_generator.py
```

Se verifica el ZIP oficial por SHA-256 y se prepara el modelo desde el registro
oficial. `data/catalog/local_generator.v1.json` fija tag, digest, cuantización,
tokenizer, revisión HF y hashes. Los pesos, runtime y modelos preparados permanecen
fuera de Git. El tag híbrido original de Qwen3 seguía produciendo razonamiento
durante las pruebas; se eligió la variante oficial Instruct sin ese modo.

La configuración se toma de `.env`, únicamente en el servidor. El selector de
generación local muestra únicamente el modelo configurado, disponible en `/api/tags`.
En el modo Cloud opcional, cuando un nombre
configurado termina en `-cloud` y el proveedor sólo anuncia su equivalente sin
ese sufijo, se usa ese equivalente exacto anunciado. No se modifica `.env`.

Para ver la respuesta o error del proveedor configurado:

```powershell
.\.venv\Scripts\python.exe scripts\check_ollama.py
```

El script consulta los modelos y hace una única petición de generación con el
texto «Responde solamente: OK», limitada a 128 tokens de salida. No envía manual,
historial ni información clínica. Muestra HTTP y cuerpo de respuesta, ocultando
la clave si el proveedor la repitiese. No imprime cabeceras de autenticación ni
hace reintentos. En local no hay coste de API; en Cloud puede consumir cuota.

Opciones:

```powershell
.\.venv\Scripts\python.exe scripts\check_ollama.py --list-only
.\.venv\Scripts\python.exe scripts\check_local_rag.py
```

## Funcionamiento

El servidor comprueba hashes del informe e índice, la identidad del PDF, forma,
finitud y normalización de las matrices. Mantiene un solo encoder cargado en la
RTX, float16 y dos hilos CPU. Cambiar de encoder libera el anterior. Rechaza una
nueva carga si hay menos de 2 GiB de RAM disponible y admite una consulta a la vez.
En Chat libera el encoder antes de cargar el generador. La generación solicita
`keep_alive=0`, de modo que Ollama descarga el modelo al terminar. Se usan 8192
tokens de contexto, hasta 512 de salida, dos hilos CPU y batch de prompt 128.
La comprobación de GPU mostró 3.721.563.996 bytes en GPU (aprox. 3,7 GB decimales),
igual al tamaño total cargado que informa Ollama: 100 % GPU.

El contexto se cuenta previamente con el tokenizer oficial de Qwen3 4B Instruct,
ChatML y margen de 128 tokens. Se reserva la salida y se rechaza el exceso sin
truncar. En la prueba fueron 2327 tokens de entrada reales y presupuesto 2455;
99 tokens de salida y finalización normal. Se contrasta el conteo con el usage real.

`OLLAMA_NO_CLOUD=1` se respeta tanto al arrancar Ollama como en el cliente Python:
se rechazan proveedores remotos. El cliente no envía Authorization a localhost,
aunque una clave Cloud antigua permanezca en `.env`. HTTP sólo se admite para
loopback; los proveedores remotos seguirían requiriendo HTTPS y activación explícita.

La pregunta se codifica con el prefijo o instrucción del encoder. Para resolver
referencias conversacionales puede añadirse la última pregunta del usuario si
cabe en la ventana. Las preguntas demasiado largas se rechazan; no se truncan.
La búsqueda exacta devuelve cuatro pasajes de la colección elegida, con páginas
físicas, etiquetas impresas, texto, offsets conservados en el índice y referencia
al padre. Las similitudes mostradas no son probabilidades de corrección.

En modo Chat el generador recibe instrucciones, hasta seis mensajes recientes
acotados y los cuatro fragmentos actuales. La respuesta debe citar [S1], [S2],
etc. Una referencia abre el texto y el PDF de esa respuesta, incluso si después
se han hecho más preguntas. Los IDs de citas se comprueban contra el contexto
actual: sin citas o con IDs inexistentes se muestra abstención.
Esta comprobación no valida que cada afirmación se deduzca de la fuente.

El control de abstención por similitud del piloto no se aplica automáticamente:
se calibró con un único caso negativo y no está validado para preguntas libres.
El generador recibe instrucciones de abstenerse cuando el texto sea insuficiente.
Esa conducta todavía requiere evaluación real y revisión humana. Las figuras no
están interpretadas y la recuperación puede fallar, como se observó en la consulta
general sobre información clínica previa: el pasaje esperado no apareció en top 4.

Las conversaciones permanecen en memoria del navegador y no se guardan en disco
ni localStorage. Los logs HTTP no registran preguntas ni respuestas. El backend
no expone rutas arbitrarias del proyecto, `.env` o código fuente. Acepta peticiones
JSON del mismo origen y limita el tamaño de entrada; los textos del modelo se
renderizan como texto y nodos DOM, sin interpretar HTML procedente del modelo.
Es un servidor de desarrollo local, no un despliegue para usuarios externos.

## Historial de validación inicial (14-11-2026)

- 84 pruebas del proyecto superadas; Ruff y sintaxis JavaScript correctos.
- Servidor probado con el usuario normal de Windows, fuera del sandbox.
- Búsqueda real con Qwen3, página 19 del PDF entre los primeros resultados para
  la consulta de selección de lesión índice.
- Probados archivos de interfaz, PDF, rutas privadas rechazadas y origen externo.
- Dos intentos reales de generación recibieron **HTTP 402** de Ollama Cloud.
  No hay una respuesta generada real validada. No se han repetido esos intentos
  mientras el usuario revisa el proveedor.
- El estado 402 queda registrado sin secretos en `output/chat_provider_status.json`.
  Al abrir la interfaz se muestra y se selecciona búsqueda local. Tras resolverlo,
  el usuario puede volver al modo Chat y enviar una nueva pregunta; un éxito borra
  ese estado de error. Listar modelos no demuestra que se disponga de cuota.
- No se pudo hacer inspección visual ni interacción de navegador: el servicio de
  acceso al navegador no está disponible. La comprobación realizada es HTTP y de
  lógica del frontend, no una prueba visual.

Las carpetas publicadas por TemporaryDirectory heredaban una DACL privada de
Windows que impedía leerlas desde el usuario normal. Se restauraron los permisos
heredados exclusivamente de los artefactos afectados. La publicación ahora usa
`artifact_stage`, con permisos normales de la carpeta de destino y limpieza
limitada al staging propio. No se modificaron controles de seguridad de Windows.
La fuente de 03 se actualizó para esa publicación; su ejecución histórica de
61,403 s se conserva con el hash de la fuente que realmente se ejecutó. No se ha
repetido la inferencia de 03 por este cambio de permisos.

La prueba Cloud inicial quedó bloqueada en el proveedor; la validación local del
15-11-2026 sí completó el recorrido, en 35,7 s con 8,1 s de generación. Evidencia:
`output/chat_validation_local.json`, `output/chat_local_example.json` y
`output/diagnostics/local_generator_smoke.json`. 89 pruebas superadas, Ruff y
sintaxis JS correctos. Al terminar no había encoder cargado ni modelo en `/api/ps`.
No se repitió la inspección visual del navegador, que seguía sin servicio accesible.
El notebook experimental 04,
incluida su comparación con documento completo y controles, sigue pendiente;
esta interfaz no sustituye ese experimento.

Referencias oficiales de la integración:
[API chat](https://docs.ollama.com/api/chat),
[Ollama Cloud](https://docs.ollama.com/cloud),
[runtime Windows](https://docs.ollama.com/windows),
[Qwen3 4B Instruct](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M).
