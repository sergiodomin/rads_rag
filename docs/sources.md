# Investigación documental y derechos

Verificado: **2026-11-11**. Los enlaces pueden cambiar: consultar la página del organismo
antes de actualizar un PDF. La fecha de publicación de una web no sustituye la versión
interna del documento. Estas notas registran condiciones observadas, no un dictamen legal.

## Fuente actual

La [página oficial ACR de PI-RADS](https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/PI-RADS)
identifica el desarrollo conjunto ACR, ESUR y AdMeTech y enlaza en «Current Release»
un PDF de **PI-RADS v2.1 (2019)** de 76 páginas. No se identificó una versión posterior
en esa página en la fecha de revisión. El host Sitecore del catálogo es el destino
directo de ese enlace oficial, no una copia de terceros.

El mismo portal enlaza el manual v2 de 2015 (64 páginas), el resumen de cambios a 2.1
(29 páginas), el léxico (9 páginas), un atlas y plantillas de informe PDF/TXT.
Se catalogan manuales, cambios y léxico; atlas interactivo y plantillas se dejan como
recursos de consulta en el portal. El manual combina prosa, tablas de criterios,
ilustraciones, mapas y apéndices. Ese inventario se verificó inicialmente en la fuente web. El 2026-11-13 se inspeccionaron
las 76 páginas del PDF aportado y se ejecutó la extracción local; la revisión humana
de las relaciones tabulares sigue pendiente.

Los [términos de ACR](https://www.acr.org/Legal) imponen condiciones de copia.
La incorporación requiere una copia autorizada para procesamiento local; el manual
no se considera de licencia abierta. `auto_download=false` impide su descarga
automática. Los originales y los registros locales de autorización quedan fuera
de Git. Está disponible el formulario oficial de
[solicitud de permiso](https://form.jotform.us/71095935087162).

## Fuente abierta histórica

**Barentsz JO et al. ESUR prostate MR guidelines 2012. European Radiology 22:746-757.**
[Editor y PDF](https://link.springer.com/article/10.1007/s00330-011-2377-y),
[DOI](https://doi.org/10.1007/s00330-011-2377-y),
[registro y licencia en PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3297750/).

Guía original de consenso de un grupo ESUR, antecedente de PI-RADS v1. El editor
declara Creative Commons Attribution Noncommercial: permite el uso no comercial con
atribución. Se conserva el PDF original, sin alterar sus avisos, y su manifiesto local.
El catálogo enlaza la declaración exacta en lugar de adivinar una versión de licencia.
El notebook muestra la atribución junto a la exploración. El editor devolvió HTML en
lugar del PDF durante la adquisición. Se obtuvo el archivo de **PMC3297750.1**, depósito oficial
Open Access de PMC, usando su [servicio Cloud documentado](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/).
Los [metadatos oficiales](https://pmc-oa-opendata.s3.amazonaws.com/PMC3297750.1/PMC3297750.1.json)
confirman DOI, título, `is_manuscript=false`, `is_retracted=false` y `license_code=CC BY-NC`.
El catálogo conserva ambos enlaces. No se scrapea la web PMC ni se usan espejos de terceros.
El antiguo servicio `oa.fcgi` ya no está disponible tras la migración de agosto de 2026.

**Separación temporal obligatoria:** esta guía de 2012 sólo permite estudiar estructura,
parsing y antecedentes. No se utilizará como evidencia de los criterios actuales de
PI-RADS 2.1. El notebook y los registros exportados incluyen versión y papel documental.

## Incorporación local realizada (2026-11-13)

`PIRADS-2019.pdf` se copió conservando el original a
`data/raw/pirads-2.1-2019/manual.pdf`. El manifiesto `.source.json` registra
`acquisition_method=user_provided_local_file`, fecha UTC, nombre aportado,
versión 2.1, 76 páginas, 3,019,107 bytes y la base de uso registrada localmente.
SHA-256: `594e618213e76205a5a91ca3df488274b0f6f42325b173581577da556cf2b4f2`.

Para reproducir la incorporación sobre una copia del mismo archivo y con la
cobertura de uso correspondiente:

```powershell
uv run radiology-rag import pirads-2.1-2019 .\PIRADS-2019.pdf --authorization-note "Cobertura de uso confirmada para incorporación y procesamiento local"
uv run radiology-rag verify
uv run radiology-rag process pirads-2.1-2019
```

Repetir la incorporación de la misma identidad verifica y reutiliza el manifiesto;
una identidad distinta se rechaza. El original y los derivados quedan excluidos de Git.
Los notebooks seleccionan `active_document_id` del catálogo; se puede fijar ESUR-2012.

El manual contiene el léxico en el apéndice III (páginas físicas 39–47), además de
criterios y ejemplos. La versión 2.0 y el resumen de cambios aportan una comparación
histórica; no son necesarios para empezar con v2.1. No se han incorporado automáticamente.

## Otras incorporaciones por descarga autorizada

1. Consultar la página ACR y confirmar versión y alcance del permiso: almacenamiento,
   extracción local y, si se prevé, envío de fragmentos a un proveedor Cloud.
2. Conservar el permiso fuera de Git. No basta cambiar `auto_download` a `true`.
3. Con autorización real, ejecutar (sustituir la descripción por una referencia local):

   ```powershell
   uv run radiology-rag download pirads-2.1-2019 --authorization-note "Referencia al permiso y alcance aplicable"
   uv run radiology-rag verify
   ```

   El texto del ejemplo no es un permiso. El comando sólo registra la base declarada;
   no puede verificar jurídicamente una autorización. Si el enlace necesita autenticación,
   no eludirla: descargar desde el portal autorizado y registrar URL, fecha, versión,
   número de páginas y SHA-256 con el mismo esquema `.source.json`, antes de explorar.
4. Seleccionar `DOCUMENT_ID = "pirads-2.1-2019"` en el notebook 01 y volver a ejecutar
   todo. Las páginas y resultados de 2012 no son transferibles al nuevo documento.

## Catálogo e integridad

BI-RADS se registra como segunda familia documental. Edición oficial v2025 y una
revisión abierta de 2026 se distinguen por sistema, edición y papel documental.
El PDF y XML de la revisión están incorporados y procesados en un índice
estructural. Véanse [fuentes BI-RADS y tratamiento de tablas](birads_sources.md).

`documents.v1.json` guarda versión del esquema y del catálogo, identificador estable,
responsable, versión documental, URL fuente, URL del archivo, formato, idioma, páginas
esperadas y política de derechos. El estado indica la posibilidad de adquisición;
la presencia local se consulta en disco, no se finge en el catálogo compartido.

Cada descarga crea un `.source.json` junto al PDF con fecha UTC, hash, bytes, páginas,
ETag/Last-Modified si existen y fundamento de uso. Un hash demuestra identidad entre
ejecuciones, no autenticidad clínica ni corrección del contenido. Una actualización
requiere un nuevo ID/registro; nunca sobrescribir el original silenciosamente.
