# Registro de aprendizaje

## 2026-11-11 · Iteración 1

**Decisiones:** repositorio inicialmente vacío; empezar con infraestructura mínima y
dos cuadernos con lógica visible. El PDF accesible de ACR no implica licencia abierta.
El usuario confirmó ausencia de autorización: manual v2.1 pendiente, guía primaria
ESUR 2012 como corpus histórico con permiso no comercial explícito.

**Hipótesis que se investigan en 01:** la extracción ordenada por coordenadas puede
alterar el orden de doble columna; una tabla puede producir texto legible y perder
relaciones; detectar pocos caracteres no prueba por sí mismo necesidad de OCR.
No se afirman mejoras de RAG: aún no existe retrieval ni conjunto de evaluación.

**Evidencia:** ver `docs/validation.md`, ejecuciones locales en `output/notebooks/` y
perfiles en `data/processed/`. Las conclusiones sobre un parser requieren contrastar
regiones con el PDF. Longitud o coincidencia entre parsers no son exactitud semántica.

**Siguiente decisión:** elegir regiones para 02 a partir de los diagnósticos medidos,
anotar una referencia manual y comparar cortes manteniendo condiciones y notas.

## 2026-11-11 · Iteración 2: parsing y chunking

**Implementación:** tokenizer E5 real fijado por commit/hash, sin ejecutar encoder;
mapa de regiones específico del PDF y diez microreferencias visuales pendientes de
revisión humana. Notebook 02 con algoritmos de corte visibles y equivalencia comprobada
contra las funciones reutilizables. Nueve configuraciones ejecutadas: fijo y estructural
en 256/512/768 tokens, overlap fijo 0/15%. Semántico disponible como celda opcional,
desactivada y sin resultados atribuidos.

**Hallazgo de parsing:** el diagrama vectorial de página 4 produce rótulos extraíbles
sin relaciones. Se excluyeron de la prosa de recuperación y se registró el objeto para
revisión. Las tres imágenes tampoco se transcribieron. Una lista de palabras no basta
para reconstruir un algoritmo radiológico. `within_bbox` evita incluir texto que sólo
interseca el borde de una microregión de referencia.

**Mediciones:** 29,209 caracteres seleccionados, 80 bloques. Fijo 512/15% produce 18
chunks, factor de copias 1.167; estructural 512 produce 26 y conserva 4/4 grupos de
tabla, pero dos padres exceden la ventana de E5. Cobertura del texto seleccionado 100%,
sin afirmar cobertura del PDF. 48 tests superados; notebook completo ejecutado.

**Interpretación:** conservar contexto y caber en el encoder son objetivos distintos.
El overlap no mejora todos los casos: desplaza fronteras y duplica contenido. Las
microreferencias son diagnósticos sesgados, no validación independiente de extracción.
No se han medido mejoras de retrieval ni respuestas de un LLM.

**Próxima decisión:** en 03, comparar embeddings y recuperación sobre estas mismas
colecciones, manteniendo un tratamiento explícito de padres largos. Diseñar hijos
con encabezados/condiciones/notas y revisar manualmente evidencia antes de llamarla gold.


## 2026-11-13 · Incorporación de PI-RADS v2.1

El usuario aportó el manual y confirmó la cobertura de uso. Se incorporó con hash,
procedencia local y originales excluidos de Git. 01 y 02 seleccionan el documento
activo del catálogo; ESUR se mantiene separado para comparación histórica.

El mapa de una columna de PI-RADS sustituye las suposiciones de doble columna de ESUR.
Se revisaron las 76 páginas; encabezados decorativos provocaban falsos candidatos de
tabla. Se distinguen 84 imágenes decorativas y 54 informativas pendientes; no se
declaran figuras transcritas. El léxico ya forma parte del manual (apéndice III).

Resultado: 224 bloques seleccionados, 85,110 caracteres. Fijo 512/15% produce 50
chunks; estructural 512 produce 55 y mantiene 14/14 grupos tabulares, con un padre
de 724 tokens. Conservar la tabla completa y caber en E5 siguen siendo problemas
distintos. Las anclas repetidas deben localizarse por página para evitar falsos éxitos.

58 pruebas superadas y notebooks ejecutados. La publicación es compartida por CLI y
02, por huella y sin reemplazar el baseline con el experimento semántico opcional.
Todavía no se calculan embeddings. Próxima decisión: tratar los padres largos y
medir recuperación bilingüe con el encoder local en 03.

## 2026-11-14 · Embeddings reales y recuperación bilingüe

**Implementación:** 03 completo en CUDA/RTX 2060, con PyTorch y Transformers
directos. E5 produce 768 dimensiones mediante media enmascarada; Qwen3 produce
1024 mediante último token válido. Los vectores normalizados y sus pasajes se
guardan con hashes, revisiones, páginas, offsets y referencia al padre. Un padre
de 724 tokens se convierte en dos hijos sin truncar ni sustituir el original.

**Mediciones:** 106 pasajes únicos por modelo y 24 consultas bilingües, también
codificadas sin instrucción. E5 tarda 1,828 s en los pasajes y Qwen3 8,964 s;
VRAM máxima asignada 555,9 y 1217,0 MiB. Notebook: 61,403 s. 65 pruebas superadas.

**Hallazgo:** en cuatro casos respondibles de test por idioma, Qwen3 con cortes
fijos alcanza EvidenceRecall@5 1,0 en ES y EN. Con estructura cae a 0,5 en ES.
E5 obtiene 0,5 en ES y 1,0 en EN en ambas colecciones. El orden del primer
acierto también cambia. La conservación de estructura no garantiza mejor ranking;
la longitud del pasaje y la agregación del encoder intervienen. Estas últimas
son hipótesis para estudiar, no causas demostradas por este piloto pequeño.

**Abstención:** ambos encoders con estructura aceptan el control visual de test
en inglés, aunque falta esa información en el texto. Los umbrales se fijaron sólo
con desarrollo. Similitud y suficiencia documental requieren comprobaciones distintas.
El conjunto sigue pendiente de revisión humana y de más controles independientes.

**Sistema:** DISM y SFC repararon daños y el usuario reinició. La prueba CUDA y
03 terminaron sin reinicio; la causa de los tres fallos previos sigue sin confirmar.

**Siguiente decisión:** revisar evidencia y controles antes de 04, donde se añadirá
el generador y el baseline RAG. El paso 03 no ha usado ningún LLM.

## 2026-11-14 · Interfaz de consulta

Se implementó un chat local con HTML/CSS/JavaScript, backend HTTP de Python e
índices de 03. Hay selección de encoder y colección, historial reciente, fuentes
por respuesta y enlaces al PDF. La clave no llega al navegador. Búsqueda local
real comprobada; 84 pruebas, Ruff y sintaxis JS correctos. No hubo acceso al
navegador para verificar visualmente la interfaz.

La integración real con el generador recibió HTTP 402 en dos intentos. La interfaz
muestra el estado y conserva búsqueda local, sin inventar respuestas. El usuario
pidió un diagnóstico manual: `scripts/check_ollama.py` hace una petición mínima,
muestra el error del proveedor y oculta la clave. No se ejecutó ese diagnóstico
automáticamente; sólo se comprobó su ayuda y la ocultación mediante una prueba.

Se corrigieron permisos de staging que impedían al usuario normal leer los
índices publicados. La publicación hereda ahora permisos del proyecto. La
evaluación experimental del generador y el notebook 04 siguen pendientes.

## 2026-11-15 · Generación local que cabe en la RTX

Se sustituyó el endpoint Cloud por Ollama portable local y Qwen3 4B Instruct
2507 Q4_K_M. Modelo real 100 % GPU, unos 3,7 GB decimales con contexto 8192.
La primera variante híbrida seguía generando razonamiento; se eligió la variante
oficial Instruct, sin modificar pesos ni entrenar otro modelo.

El encoder se libera antes de cargar el generador y Ollama lo descarga al terminar.
Así no se mantienen ambos modelos simultáneamente en VRAM. Se usa el tokenizer
del generador fijado por revisión/hash y se reserva salida antes de generar.
El exceso de contexto se rechaza, no se trunca silenciosamente.

Una consulta RAG real respondió en español con [S2], 2327 tokens de entrada y
99 de salida: 35,7 s completos y 8,1 s de generación. Cero llamadas Cloud,
89 pruebas superadas. La prueba de funcionamiento no constituye evaluación clínica
ni explica la causa de los antiguos reinicios. Próximo experimento: 04, con un
conjunto y controles de generación definidos; el documento completo sólo se
comparará si cabe realmente con su margen de salida.

## 2026-11-15 · Repositorio público y siguientes arquitecturas

Se preparó una distribución de código, catálogos, notebooks sin salidas y
documentación. Se excluyeron también las anotaciones con transcripciones y anclas
del manual: poder usar el PDF localmente no implica redistribuirlo ni publicar
sus extractos de evaluación. El procesamiento admite referencias ausentes y hay
un constructor de índices sin evaluación para una instalación nueva.

Se documentó la diferencia entre recuperación densa, contexto completo, CAG y
GraphRAG. El corpus seleccionado suma 19.197 tokens con el tokenizer del generador,
sin instrucciones ni salida: no cabe en 8192. CAG requiere un subconjunto y
verificar reutilización de caché; el chat actual descarga el modelo al terminar.

La consulta multi-RADS requiere cambios de catálogo, filtros, índices y citas
por sistema y versión; todavía sólo PI-RADS v2.1 está disponible en el chat.
La búsqueda español/inglés utiliza embeddings multilingües, sin traductor.
92 pruebas y Ruff superados. Los diseños futuros no se presentan como resultados.
