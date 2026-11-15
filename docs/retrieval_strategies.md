# Recuperación densa, CAG, grafos y varios RADS

Estado a 2026-11-15. El sistema ejecutado es un RAG denso sobre PI-RADS v2.1
y una revisión secundaria de BI-RADS v2025 publicada en 2026.
CAG, GraphRAG y la consulta simultánea de varios RADS son propuestas de experimentos.

## Qué significa denso

Un encoder convierte una consulta o un pasaje en un vector aprendido: 1024 números
en Qwen3 Embedding 0.6B o 768 en E5 multilingual base. «Denso» describe esa
representación compacta, con componentes normalmente distintas de cero. No indica
la cuantización del generador ni cómo se han cortado los documentos.

Normalizamos los vectores y calculamos su producto escalar, equivalente al coseno.
La búsqueda exacta NumPy elige los cuatro pasajes mejor puntuados de la colección
seleccionada. El generador recibe pregunta, pasajes y referencias de página/versión.
No se entrenan nuevos pesos. BM25 y recuperación híbrida aún no están implementados.

En recuperación léxica dispersa, dimensiones o términos reflejan coincidencias de
vocabulario. La recuperación densa puede asociar formulaciones semánticamente
relacionadas sin compartir palabras; también puede confundir criterios cercanos.
Un número alto de similitud no demuestra que la evidencia sea suficiente.

## Español e inglés

El corpus actual está en inglés. El encoder multilingüe proyecta la pregunta
española y los pasajes ingleses al mismo espacio de vectores. **No hay traducción
intermedia ni copia española del manual**. E5 usa `query:` y `passage:`; Qwen usa
la instrucción de recuperación configurada para consultas. Esos formatos no
traducen la pregunta. El generador recibe el texto original y responde en español.

Los tokenizers tampoco son traductores: segmentan cadenas en IDs. Para cortar
se usa el tokenizer E5, para cada forward el del encoder correspondiente y para
presupuestar el prompt el de Qwen3 4B Instruct, fijado por revisión y hashes.

La capacidad multilingüe no garantiza equivalencia ES/EN. El piloto actual tiene
12 casos y 24 consultas; requiere revisión humana. Una futura ablación puede
comparar consulta original, traducción explícita y combinación de candidatos,
conservando la consulta original y midiendo errores en negaciones, unidades y
términos radiológicos. No se añadirá ese paso sin medir su aportación.

## Comparación de arquitecturas

| Método | Qué recibe el generador | Trabajo adicional | Uso a comprobar |
|---|---|---|---|
| RAG denso actual | Pasajes recuperados por consulta | Embeddings e índice | Preguntas documentales localizadas |
| Contexto completo | Corpus o subconjunto completo en cada prompt | Prefill del contexto | Control para medir lo que pierde la recuperación |
| CAG | Corpus estable en un prefijo con estados reutilizables | Construcción y conservación de caché KV | Consultas repetidas sobre un corpus pequeño y fijo |
| KG-RAG | Texto y relaciones recuperadas de un grafo | Esquema, extracción y revisión | Preguntas que requieren enlazar entidades y condiciones |
| Microsoft GraphRAG | Evidencia del grafo y/o resúmenes de comunidades | Extracción con LLM, grafo, comunidades y resúmenes | Preguntas globales y síntesis transversal |

Son diseños que pueden combinarse. Un índice de embeddings cacheados no constituye
por sí mismo CAG. Poner un documento entero en un prompt tampoco demuestra ahorro
por caché: hay que comprobar qué estados reutiliza realmente el motor de inferencia.

### Experimento CAG propuesto

El [artículo CAG](https://arxiv.org/abs/2412.15605) propone precargar conocimiento
manejable en contextos largos y reutilizar estados de atención. Nuestra medición
del texto seleccionado de PI-RADS da **19.197 tokens** con el tokenizer fijado del
generador, sin truncación y antes de añadir instrucciones, pregunta y respuesta.
La ventana configurada es **8192**, con hasta 512 tokens de salida. Por tanto,
el corpus completo seleccionado no cabe en la configuración actual.

Probar primero un subconjunto coherente que quepa con margen y comparar sobre
las mismas preguntas: RAG, contexto completo sin caché reutilizada y CAG con
caché verificada. Medir preparación inicial, latencia de primera consulta y
consultas repetidas, tokens realmente procesados, VRAM/RAM, fidelidad y citas.
Separar preguntas respondibles dentro del subconjunto de las que quedan fuera.

El chat actual descarga el generador con `keep_alive=0` al acabar para limitar
la memoria residente. Ese comportamiento impide asumir una caché KV persistente
entre consultas. Un experimento CAG necesitará una política explícita de residencia,
un prefijo estable y evidencia del runtime sobre reutilización; no basta cambiar
el nombre del endpoint. Actualizar el documento o su orden puede invalidar cachés.
La ventana máxima anunciada por el modelo tampoco garantiza que quepa en 6 GiB.

### Experimento con grafo propuesto

Empezar con un grafo pequeño revisado: sistema, versión, anatomía, secuencia,
hallazgo, condición, categoría y recomendación. Cada relación debe conservar su
pasaje y página de origen, versión, condiciones, excepciones y negaciones.
Una relación extraída por un LLM es una hipótesis hasta revisarla.

El [artículo de Microsoft GraphRAG](https://arxiv.org/abs/2404.16130) estudia
extracción de entidades y relaciones y resúmenes de comunidades para preguntas
globales. Su resultado no demuestra superioridad sobre preguntas precisas de
criterios PI-RADS. Comparar nuestro baseline y el grafo en preguntas multisección
y globales, midiendo cobertura, errores de relaciones, fidelidad y coste de
construcción/actualización. Los resúmenes generados no sustituyen a la fuente.

## Diseño multi-RADS

El registro `chat_document_ids` habilita documentos con índices completos. El chat
incluye PI-RADS v2.1 y una revisión secundaria de BI-RADS v2025, con filtros por
sistema y citas por documento. ESUR-2012 permanece como antecedente separado.
El manual oficial BI-RADS, LI-RADS y TI-RADS no están incorporados. Una entrada
en el catálogo no habilita una fuente sin su procesamiento e índice verificables.

El diseño general y los requisitos para futuras ampliaciones comprenden:

1. Registrar por documento `system`, `document_id`, versión, idioma, órgano/modalidad,
   organismo, procedencia, derechos y SHA-256. Revisar mapas de parsing y tablas
   por documento. «Vigente» debe ser una decisión explícita revisada.
2. Crear índices por documento y versión usando el mismo encoder/revisión y
   contrato de normalización. No comparar directamente vectores de modelos distintos.
3. Sustituir la selección única por filtros de sistema/versión y una opción de
   consultar todos los corpus activos. Recuperar candidatos y ordenar conjuntamente
   con presupuestos que eviten que un manual grande eclipse a los demás.
4. En preguntas ambiguas, pedir órgano/sistema. Para preguntas cruzadas, recuperar
   evidencia de cada familia y citar sistema, versión y página en cada afirmación.
   No equiparar una categoría numérica entre sistemas distintos.
5. Habilitar versiones históricas de forma explícita y advertir discrepancias;
   no mezclar criterios antiguos y actuales en una única respuesta sin atribución.
6. Evaluar cada familia y consultas cruzadas: ruta correcta, cobertura, unidades,
   negaciones, conflictos de versiones, citas y abstención. Las condiciones de uso
   de un documento no se heredan por compartir el sufijo RADS.

Un router puede sugerir filtros, pero no debe ocultar silenciosamente fuentes
necesarias. La clasificación clínica de un caso requeriría reglas y validación
adicionales; queda fuera de la consulta documental de este laboratorio.

## Orden de trabajo recomendado

Primero ampliar el registro y los filtros multi-corpus con versiones y evaluación.
Después comparar BM25 + dense con RRF y, si aporta valor, reranking. Hacer CAG
como experimento acotado y medido. Construir un grafo cuando exista un conjunto de
preguntas relacionales que justifique su coste. Estas decisiones son hipótesis del
laboratorio, no resultados ya demostrados.

Referencias de los encoders: [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
y [multilingual E5 base](https://huggingface.co/intfloat/multilingual-e5-base).
