# Validación técnica y límites

Estado: **2026-11-15**, experimental y sin validación clínica.

## Comprobaciones de software

La suite comprende **195 pruebas** de configuración, procedencia, segmentación,
recuperación, fusión, reranking, presupuesto, contratos HTTP y validación de citas.
Se comprueban también Ruff, sintaxis JavaScript y archivos publicables.

```powershell
uv run ruff check src tests scripts
uv run pytest -q
node --check src/radiology_rag/web/static/app.js
node --check src/radiology_rag/web/static/lab.js
uv run python scripts/check_publish.py
# Tras preparar el commit:
uv run python scripts/check_publish.py --staged
```

La auditoría comprueba rutas excluidas, tamaños y patrones de secretos; no sustituye
la revisión del contenido ni garantiza ausencia de cualquier dato sensible.

## Comprobaciones locales del pipeline

Se ejecutaron consultas bilingües, categorías tabulares, comparaciones entre
sistemas y controles con y sin reranking. Los modelos se cargan por turnos.

Una consulta mixta nativa recuperó 20 candidatos textuales y 2 visuales, reordenó
los 22 y seleccionó 3 textos y 1 PNG. La respuesta usó IDs por modalidad sin
truncación. Otro control completó la ruta común con 4 textos y ninguna imagen.
Son pruebas de funcionamiento, no mediciones de calidad general.

Se revisaron los cuatro recortes registrados y su geometría. Se comprobaron las
rutas HTTP del chat y de las imágenes, junto con los hashes de los PNG.
Estas comprobaciones no equivalen a una revisión visual completa de la interfaz.
Las trazas y resultados detallados se conservan localmente, fuera de Git.

## Límites conocidos

- En un piloto de seis consultas con referencias provisionales, el reranking pasó
  de recall medio de evidencia 0,5833 a 0,5. No se puede concluir una mejora general.
- Se observaron fallos al interpretar relaciones de diagramas y al atribuir
  afirmaciones a fuentes. Un ID válido no demuestra fidelidad semántica.
- El baseline de campos de alcance sólo cubre los campos registrados; no es un
  sistema general de verificación de respuestas.
- Las referencias requieren revisión independiente. Los benchmarks públicos
  discutidos no se han incorporado ni evaluado en este proyecto.
- No se ha demostrado utilidad clínica ni ahorro de tiempo para radiólogos.
