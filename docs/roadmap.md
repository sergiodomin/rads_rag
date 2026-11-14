# Roadmap educativo y criterios de avance

Cada cuaderno debe expresar objetivo, fundamento, implementación visible, experimentos,
interpretación y conclusiones basadas en ejecuciones. Mantener una tabla de hipótesis,
configuraciones y resultados, con entorno y hash del corpus. No crear archivos vacíos
para fases futuras. Implementados: 00, 01, 02 y 03. En 02 se han ejecutado parsing,
chunking fijo y estructural; la sección semántica es opcional y no se ha ejecutado.
03 se ejecutó completo el 14-11-2026 en la RTX 2060: E5 y Qwen3, índices NumPy,
recuperación bilingüe, costes y controles de abstención. [Resultados](embeddings.md).

| Etapa | Pregunta e implementación | Experimento y criterio de salida |
|---|---|---|
| 00, ahora | ¿Qué significa una respuesta trazable? Catálogo, versiones, configuración y protocolo | Auditar permisos, hash, separación proveedor/embeddings y un registro experimental real |
| 01, ahora | ¿Qué conserva y qué pierde un parser PDF? Inventario, texto, coordenadas, tablas y render | Comparar pypdf y pdfplumber página a página; elegir páginas difíciles y exportar perfiles con procedencia |
| 02, implementado | ¿Dónde cortar sin separar criterios de sus condiciones? Parsing y chunking | 10 microregiones con revisión visual del agente (humana pendiente); fixed 256/512/768 tokens, overlap 0/15% y estructural. Contención, duplicación, cobertura seleccionada, longitud y coste medidos. Semántico opcional pendiente; no se inventan resultados |
| 03, ejecutado | ¿Cómo afecta el idioma a la recuperación? Embeddings locales y similitud desde NumPy | E5 multilingual base frente a Qwen3 Embedding 0.6B con revisiones y formatos fijados. 12 casos / 24 consultas ES/EN, revisión humana pendiente. Recall@k, MRR, evidencia, costes y controles medidos; 50 pasajes fijos frente a 56 estructurales. Sin truncación silenciosa; búsqueda exacta |
| 04 | ¿Qué aporta recuperar frente a dar todo el documento? Baseline RAG | Prompt verificable con citas página/versión y abstención. Comparar RAG, documento completo si cabe con margen de salida y respuesta sin contexto como control. Congelar modelo/opciones; medir citas, fidelidad, abstención, tokens y latencia. Nunca truncar silenciosamente el documento completo |
| 05 | ¿Ayuda la coincidencia léxica de términos técnicos? BM25 desde sus ecuaciones, dense y RRF | Misma evaluación, k y presupuesto de candidatos. RRF: suma 1/(c+r); variar c y pesos sin ajustar con test. Inspeccionar consultas donde cada recuperador falla |
| 06 | ¿Vale el coste de comparar pares query-passage? Cross-encoder | Congelar candidatos y rerankear; separar recall de candidatos del orden final. Medir nDCG/MRR y latencia. Reranker distinto del generador. Comparar presupuesto de cómputo y errores de idioma |
| 07 | ¿Podemos confiar en las métricas? Evaluación formal y revisión | Consolidar conjunto iniciado en 02, separar dev/test, etiquetas de relevancia graduada y revisión documentada. Recall, Precision, MRR, nDCG deterministas; fidelidad y exactitud documental con revisión y juez LLM calibrado. Registrar desacuerdo del juez |
| 08 | ¿Qué fallo concreto justifica otra técnica? Advanced RAG | Ablaciones de rewriting, multi-query, parent-document, contextual/hierarchical y adaptación. Cambiar una variable por vez; conservar consulta original y evidencia. El texto generado como contexto no adquiere autoridad documental. Medir mejora y coste por tipo de pregunta |
| 09 | ¿Existen preguntas relacionales que el baseline no resuelva? KG-RAG y comparación de implementaciones | Empezar con un grafo pequeño revisado, entidades/relaciones y procedencia de cada arista. Comparar sobre mismas preguntas multihop y presupuesto. Sólo después evaluar Microsoft GraphRAG, LightRAG o KAG, con costes de construcción y errores de extracción |

## Plan concreto de 02

Entrada: PDF + manifiesto; 02 funciona sin ejecutar 01. Desde el 2026-11-13 el
manual PI-RADS v2.1 aportado por el usuario es el predeterminado. Tiene mapa y
referencias propios, ejecutados; ESUR-2012 permanece disponible por separado.
Rehacer la selección de regiones si cambia el corpus. Crear una referencia de extracción revisada, distinta de las salidas
de los parsers. Las tablas deben mantener unidades, notas, condiciones y relación entre
filas/columnas. La ventana de 512 tokens de E5 obliga a medir longitud con su tokenizer,
no dividir por un número fijo de caracteres. Para comparar 768 tokens, registrar la
truncación de E5 y distinguir este efecto de la calidad de chunking; usar un encoder de
ventana mayor o limitar una comparación controlada a longitudes compatibles.

Implementar primero cortes fijos visibles, luego estructura y finalmente corte semántico
como experimento opcional, con umbral ajustado en desarrollo. No basta minimizar tamaño
de chunks: medir si una pregunta conserva todas las condiciones de su evidencia.
Salida: chunks versionados + tabla comparativa + casos revisados + hipótesis para 03.

## Evaluación desde el principio

Diseñar un piloto de preguntas directas, multisección, criterios, insuficiencia, fuera
de dominio y conflictos de versiones sólo cuando ambas versiones estén autorizadas.
Cada registro: ID, pregunta/idioma/tipo, versión, evidencia, respuesta de referencia,
abstención esperada, autor/revisor y estado. Preguntas propuestas por LLM siguen siendo
`unreviewed`. No llamarlas gold standard validado.

Reportar métricas por tipo y globales, tamaño de muestra, semillas, versiones y hash.
Precision@k depende de la definición de relevancia; nDCG requiere etiquetas graduadas.
La tasa de abstención correcta necesita ejemplos respondibles y no respondibles para
detectar abstención excesiva. Calibrar el juez LLM con una muestra humana y registrar
su modelo, prompt y desacuerdo. Evaluar citas a nivel de afirmación, no sólo contar URLs.

## Grafos y KAG: decisiones distintas

- **KG-RAG**: familia de enfoques que recupera entidades y aristas; se puede comenzar
  con estructuras en memoria y relaciones revisadas, sin servicio externo.
- **Microsoft GraphRAG**: extracción de grafo, comunidades y resúmenes para búsqueda
  local/global. La indexación puede consumir muchas llamadas LLM; no es sólo vector + vecinos.
- **LightRAG (HKUDS)**: implementación específica con índices de texto/grafo y
  recuperación de distintos niveles. Comparar su operación y costes, no asumir que
  el nombre demuestra eficiencia en este corpus.
- **KAG (OpenSPG)**: framework concreto con esquema y razonamiento guiado por formas
  lógicas; añade requisitos de modelado/operación diferentes de los anteriores.
- **RadLex (RSNA)**: candidato para normalizar entidades radiológicas; revisar licencia,
  cobertura de conceptos prostáticos, idioma y mapeos. No asumir que la ontología contiene
  las reglas de clasificación de cada versión PI-RADS.

Revisar documentación y madurez de estas herramientas al llegar a 09; no se instalan
ahora. Ninguna se considera equivalente a las otras ni superior sin resultados locales.

## Expansión documental y funcional futura

Prioridad solicitada el 2026-11-15: permitir consultar varias familias RADS.
Antes de añadir corpus, implementar registro por sistema/versión, filtros,
recuperación conjunta y citas con identidad documental. CAG se comparará con un
subconjunto: los 19.197 tokens del corpus actual no caben en la ventana de 8192.
El chat con generador local está operativo; el experimento formal 04 sigue pendiente.
Véase [el diseño detallado y las diferencias entre métodos](retrieval_strategies.md).

Incorporar BI-RADS, LI-RADS, Lung-RADS, O-RADS, TI-RADS, Bosniak, RECIST u otros sólo
con investigación propia de organismo, versión y derechos; no heredar permisos de PI-RADS.
La consulta documental exige citas verificables. La sugerencia de clasificación queda
fuera de esta iteración y requerirá extracción estructurada, reglas explícitas,
verificación de cada criterio y abstención ante datos faltantes. No se ofrece como
decisión clínica validada.
