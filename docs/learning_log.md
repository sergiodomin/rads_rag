# Decisiones técnicas

Cada documento conserva sistema RADS, versión, papel documental y SHA-256.
PI-RADS v2.1, ESUR-2012 histórico y la revisión secundaria BI-RADS permanecen
identificables. La revisión no sustituye al manual oficial BI-RADS.

Se comparan ventanas de tokens y cortes estructurales. Las tablas conservan
encabezados y notas; los padres largos pueden dividirse para el encoder sin
reemplazar el original. JATS aporta estructura explícita. Extraer rótulos de
un diagrama no reconstruye sus relaciones gráficas.

Cada encoder utiliza su tokenizer e índice independiente. La búsqueda combina
coseno, BM25 y RRF con filtros por sistema y versión. El reranker ordena textos
y representaciones textuales de figuras; no interpreta píxeles. Relevancia y
suficiencia documental requieren comprobaciones diferentes.

El chat fusiona candidatos textuales y visuales antes de seleccionar el contexto.
Qwen3-VL 2B recibe textos y un PNG cuando se selecciona una imagen. Los IDs de
cita se validan por modalidad; esto no verifica cada afirmación. Qwen3 4B
permanece como baseline textual del laboratorio.

Los grupos de desarrollo y prueba, snapshots y trazas permiten comparar variantes.
Las referencias iniciales requieren revisión independiente. Las pruebas de
software no constituyen validación clínica. Véanse [validación](validation.md),
[laboratorio](laboratory.md) y [roadmap](roadmap.md).
