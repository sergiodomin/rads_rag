"use strict";
const $ = id => document.getElementById(id);
const state = {commonGenerator: null, history: [], visualContext: "", busy: false, ready: false, sources: [], cited: [], local: false, visual: {assets: []}};
const phases = {loading_encoder: "Preparando el modelo local…", retrieving: "Buscando en los documentos…", reranking: "Seleccionando las fuentes con el reranker…", generating: "Redactando con las fuentes…"};
phases.visual_retrieving = "Buscando una figura o tabla…";
phases.visual_generating = "Analizando el contexto con el modelo común…";

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "No se pudo completar la consulta.");
  return data;
}

function toggleSources(open) {
  $("sources-panel").hidden = !open;
  $("toggle-sources").setAttribute("aria-expanded", String(open));
}

function showSources(sources, cited = [], selected = null) {
  state.sources = sources;
  state.cited = cited;
  $("source-count").textContent = String(sources.length);
  $("sources-list").replaceChildren();
  for (const source of sources) {
    const card = element("details", "source-card");
    card.id = `source-${source.id}`;
    const summary = element("summary");
    const labels = element("div", "source-labels");
    labels.append(element("span", "source-id", source.id));
    if (cited.includes(source.id)) labels.append(element("span", "cited-tag", "CITADA"));
    if (source.exact_category_match) labels.append(element("span", "cited-tag", `Categoría ${source.assessment_categories.join(", ")}`));
    summary.append(labels, element("strong", "", source.document));
    const location = source.pages.length ? `Página física: ${source.pages.join(", ")}`
      : `${(source.sections || []).map(s => s.join(" > ")).join("; ")} · ${source.xml_element_id || ""}${source.table_row ? ` · fila ${source.table_row}` : ""}`;
    const role = source.document_role === "secondary_narrative_review" ? "Revisión secundaria" : "Manual";
    const cosine = Number.isFinite(source.score) ? ` · coseno ${source.score.toFixed(3)}` : "";
    const ranking = source.image_url ? " · evidencia visual" : Number.isFinite(source.rerank_score) ? ` · reranker ${source.rerank_score.toFixed(3)}` : " · orden híbrido";
    summary.append(element("span", "source-meta", `${source.system} · ${source.document_version || ""} · ${role} · ${location}${cosine}${ranking}`));
    summary.append(element("p", "source-preview", source.text.slice(0, 150).replace(/\s+/g, " ") + "…"));
    const link = element("a", "source-pdf", source.pages.length ? "Abrir página en el PDF ↗" : "Abrir publicación original ↗");
    link.href = source.pdf_url;
    link.target = "_blank";
    link.rel = "noopener";
    card.append(summary);
    if (source.image_url) {
      const open = element("a", "source-image-link"); open.href = source.image_url; open.target = "_blank"; open.rel = "noopener";
      const image = element("img", "source-image"); image.src = source.image_url; image.alt = source.title || "Recorte del documento"; image.loading = "lazy";
      open.append(image); card.append(open, element("p", "source-meta", `Recorte original · página ${source.pages[0]} · bbox ${source.bbox.join(", ")}. Abrir imagen para ampliar.`));
      if (source.page_image_url) {
        const context = element("details", "page-context"); context.append(element("summary", "Ver región en la página completa"));
        const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
        const [width, height] = source.page_size_points;
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Página original con la región consultada resaltada");
        const page = document.createElementNS(ns, "image"); page.setAttribute("href", source.page_image_url); page.setAttribute("width", width); page.setAttribute("height", height); svg.append(page);
        const [left, top, right, bottom] = source.bbox, box = document.createElementNS(ns, "rect");
        for (const [key, value] of Object.entries({x: left, y: top, width: right-left, height: bottom-top, fill: "none", stroke: "#0086c9", "stroke-width": 3})) box.setAttribute(key, value);
        svg.append(box); context.append(svg); card.append(context);
      }
      card.open = true;
    }
    if (source.table_headers?.length && source.table_headers.length === source.table_cells?.length) {
      const table = element("table", "source-table"), body = element("tbody");
      source.table_headers.forEach((header, i) => { const row = element("tr"); row.append(element("th", "", header), element("td", "", source.table_cells[i])); body.append(row); });
      table.append(body); card.append(table);
    }
    card.append(element("p", "source-text", source.text), link);
    $("sources-list").append(card);
    if (source.id === selected) {
      card.open = true;
      card.classList.add("highlight");
    }
  }
  toggleSources(true);
  if (selected) requestAnimationFrame(() => $(`source-${selected}`)?.scrollIntoView({block: "nearest", behavior: "smooth"}));
}

function inlineText(node, text, data) {
  for (const token of text.split(/(\[S\d+\]|\*\*[^*]+\*\*)/g)) {
    if (/^\[S\d+\]$/.test(token) && data.sources.some(s => s.id === token.slice(1, -1))) {
      const button = element("button", "citation", token.slice(1, -1));
      button.type = "button";
      button.title = `Consultar fuente ${token.slice(1, -1)}`;
      button.addEventListener("click", () => showSources(data.sources, data.cited_sources, token.slice(1, -1)));
      node.append(button);
    } else if (token.startsWith("**") && token.endsWith("**")) {
      node.append(element("strong", "", token.slice(2, -2)));
    } else node.append(document.createTextNode(token));
  }
}

function renderAnswer(body, data) {
  let list = null;
  for (const line of data.answer.split("\n")) {
    if (!line.trim()) { list = null; continue; }
    const bullet = line.match(/^\s*(?:[-*]|\d+[.)])\s+(.+)$/);
    if (bullet) {
      if (!list) { list = element("ul"); body.append(list); }
      const item = element("li"); inlineText(item, bullet[1], data); list.append(item);
    } else {
      list = null;
      const heading = line.match(/^#{1,4}\s+(.+)$/);
      const paragraph = element(heading ? "h3" : "p");
      inlineText(paragraph, heading ? heading[1] : line, data); body.append(paragraph);
    }
  }
}

function assistantShell() {
  const turn = element("article", "turn assistant-turn");
  const header = element("div", "assistant-header");
  header.append(element("span", "assistant-avatar", "r."), element("span", "", "Radiology Lab"));
  turn.append(header);
  $("messages").append(turn);
  return turn;
}

function recentHistory() {
  const history = state.history.slice(-6).map(m => ({role: m.role, content: m.content.slice(0, 4000)}));
  while (history.reduce((n, m) => n + m.content.length, 0) > 10000) history.shift();
  return history;
}

function updateMode() {
  const search = $("mode").value === "search";
  for (const id of ["model", "encoder", "collection", "reranker"]) {
    $(id).hidden = false;
    document.querySelector(`label[for="${id}"]`).hidden = false;
  }
  $("visual-asset").hidden = true;
  document.querySelector('label[for="visual-asset"]').hidden = true;
  $("visual-info").hidden = search;
  $("visual-info").textContent = "El chat combina textos, tablas y figuras recuperadas con un generador común.";
  $("model").disabled = search || state.busy || Boolean(state.commonGenerator);
  $("cloud-note").textContent = search ? "Búsqueda local de fragmentos; sin generación."
    : "El chat selecciona la evidencia y envía al mismo modelo el texto y las imágenes pertinentes.";
  $("send").disabled = state.busy || !state.ready || (!search && !$("model").value);
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || state.busy) return;
  state.busy = true;
  updateMode();
  ["encoder", "collection", "reranker", "visual-asset", "system", "mode", "new-chat"].forEach(id => { $(id).disabled = true; });
  $("welcome").hidden = true;
  const user = element("article", "turn user-turn");
  user.append(element("div", "user-bubble", question));
  $("messages").append(user);
  $("question").value = "";
  const turn = assistantShell();
  const pending = element("p", "pending-text");
  const progress = element("span", "", "Buscando en el manual…");
  pending.append(element("span", "spinner"), progress);
  turn.append(pending);
  turn.scrollIntoView({block: "nearest", behavior: "smooth"});
  const poll = setInterval(async () => {
    try { const status = await api("/api/status"); progress.textContent = phases[status.phase] || "Preparando la consulta…"; }
    catch { /* La consulta principal presenta cualquier fallo de conexión. */ }
  }, 2000);
  try {
    const search = $("mode").value === "search";
    const data = await api(search ? "/api/search" : "/api/chat", {
      message: question, history: recentHistory(), encoder: $("encoder").value,
      collection: $("collection").value, model: $("model").value, system: $("system").value,
      reranker: $("reranker").value, visual_asset: $("visual-asset").value,
      visual_context: state.visualContext
    });
    pending.remove();
    if (search) {
      data.answer = `He recuperado ${data.sources.length} fragmentos. Puedes consultar sus documentos y ubicaciones en el panel de fuentes.`;
      data.cited_sources = [];
    }
    const body = element("div", "answer-body");
    renderAnswer(body, data); turn.append(body);
    const footer = element("div", "answer-footer");
    const seconds = data.metrics.total_seconds ?? data.metrics.retrieval_seconds + (data.metrics.latency_seconds || 0);
    footer.append(element("span", "", search ? "Búsqueda local · sin generación" : data.metrics.model));
    footer.append(element("span", "", `${seconds.toFixed(1)} s · ${data.sources.length} fuentes`));
    if (data.metrics.reranking?.pairs) footer.append(element("span", "badge", `Reranking · ${data.metrics.reranking.pairs} candidatos`));
    if (data.metrics.images_sent > 0) footer.append(element("span", "badge", "Lectura visual · sin revisión experta"));
    if (data.routing) footer.append(element("span", "badge", ({text: "Evidencia textual y tablas", image: "Imagen seleccionada automáticamente", combined: "Imagen y texto asociado", clarification: "Falta evidencia visual"})[data.routing.modality]));
    if (data.sources.some(source => source.kind === "flow_diagram")) footer.append(element("span", "badge", "Diagrama: comprobar flechas y columnas en el original"));
    if (data.abstained) {
      const labels = {
        insufficient_evidence: "Evidencia insuficiente",
        invalid_or_missing_citations: "Respuesta descartada: citas no válidas",
        unsupported_anatomical_term: "Respuesta descartada: órgano sin respaldo",
        missing_comparison_sources: "Faltan citas de ambos sistemas",
        unverified_scope_fields: "Faltan campos de alcance verificados"
      };
      footer.append(element("span", "badge", labels[data.abstention_reason] || "Evidencia insuficiente"));
    }
    if (data.truncated) footer.append(element("span", "badge", "Salida recortada por el límite de generación"));
    if (data.answer_mode === "verified_scope_fields") footer.append(element("span", "badge", "Comparación estructurada de fuentes"));
    if (data.metrics.query_corrections?.length) {
      footer.append(element("span", "badge", `Consulta interpretada: ${data.metrics.normalized_question}`));
    }
    const show = element("button", "", "Ver fuentes ↗");
    show.type = "button";
    show.addEventListener("click", () => showSources(data.sources, data.cited_sources));
    const copy = element("button", "", "Copiar"); copy.type = "button";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(data.answer); copy.textContent = "Copiado"; }
      catch { copy.textContent = "No se pudo copiar"; }
    });
    footer.append(show, copy); turn.append(footer);
    if (!search) { state.history.push({role: "user", content: question}, {role: "assistant", content: data.answer}); state.visualContext = data.routing?.context || ""; }
    showSources(data.sources, data.cited_sources);
  } catch (error) {
    pending.remove(); turn.append(element("p", "error-text", error.message));
    try {
      const status = await api("/api/status");
      $("provider-message").textContent = status.provider_error || "";
    } catch { /* Mantener el error original de la consulta. */ }
    const edit = element("button", "subtle-button", "Volver a editar la pregunta"); edit.type = "button";
    edit.addEventListener("click", () => { $("question").value = question; $("question").focus(); });
    turn.append(edit);
  } finally {
    clearInterval(poll); state.busy = false;
    ["encoder", "collection", "reranker", "visual-asset", "system", "mode", "new-chat"].forEach(id => { $(id).disabled = false; });
    updateMode(); $("question").focus();
  }
}

$("chat-form").addEventListener("submit", sendQuestion);
$("question").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); if (!$("send").disabled) $("chat-form").requestSubmit();
  }
});
$("mode").addEventListener("change", () => { state.history = []; state.visualContext = ""; updateMode(); });
$("model").addEventListener("change", updateMode);
let documents = [];
function updateVisualAssets() {
  const previous = $("visual-asset").value;
  $("visual-asset").replaceChildren(new Option("Selección automática", ""));
  for (const asset of state.visual.assets || []) if ($("system").value === "all" || asset.system === $("system").value) {
    $("visual-asset").add(new Option(`${asset.system} · ${asset.title} · p. ${asset.page_number}`, asset.id));
  }
  $("visual-asset").value = [...$("visual-asset").options].some(o => o.value === previous) ? previous : "";
}
function updateCollections() {
  const selected = documents.filter(d => $("system").value === "all" || d.system === $("system").value);
  for (const option of $("collection").options) {
    option.disabled = !selected.every(d => d.collections.includes(option.value));
  }
  if ($("collection").selectedOptions[0].disabled) $("collection").value = "structural_512";
}
$("system").addEventListener("change", () => { state.history = []; state.visualContext = ""; updateCollections(); updateVisualAssets(); });
$("toggle-sources").addEventListener("click", () => toggleSources($("sources-panel").hidden));
$("close-sources").addEventListener("click", () => toggleSources(false));
$("new-chat").addEventListener("click", () => {
  state.history = []; state.visualContext = ""; state.sources = []; state.cited = [];
  $("messages").replaceChildren(); $("sources-list").replaceChildren();
  $("source-count").textContent = "0"; $("welcome").hidden = false;
  $("question").value = ""; toggleSources(false); $("question").focus();
});
document.querySelectorAll("[data-question]").forEach(button => button.addEventListener("click", () => {
  $("question").value = button.dataset.question; $("question").focus();
}));

async function initialize() {
  try {
    const status = await api("/api/status");
    documents = status.documents;
    state.visual = status.visual || {assets: []};
    state.commonGenerator = status.common_generator || null;
    $("library-summary").textContent = `${documents.length} documentos · versiones identificadas`;
    for (const system of new Set(documents.map(d => d.system))) {
      const option = element("option", "", system); option.value = system; $("system").append(option);
    }
    $("document-links").replaceChildren();
    for (const doc of documents) {
      const label = doc.role === "secondary_narrative_review" ? "revisión 2026" : "manual";
      const link = element("a", "document-link", `${doc.system} · ${label} ↗`);
      link.href = `/documents/${doc.id}.pdf`; link.target = "_blank"; link.rel = "noopener";
      $("document-links").append(link);
    }
    updateCollections();
    updateVisualAssets();
    for (const reranker of status.rerankers || []) {
      const option = element("option", "", reranker.id); option.value = reranker.id; $("reranker").append(option);
    }
    $("reranker").value = status.chat_retrieval?.reranker || "none";
    state.local = status.provider_kind === "local";
    document.querySelector("label[for=model] .cloud-label").textContent = state.local ? "LOCAL" : "CLOUD";
    $("model").replaceChildren();
    for (const name of status.models) {
      const option = element("option", "", name); option.value = name; $("model").append(option);
    }
    if (status.models.includes(status.model)) $("model").value = status.model;
    else if (status.models.length) $("model").value = "";
    else { const option = element("option", "", "Configura Ollama en .env"); option.value = ""; $("model").append(option); }
    if (state.commonGenerator) {
      $("model").replaceChildren(new Option(state.commonGenerator, state.commonGenerator));
    } else if (!status.generation_ready) $("mode").value = "search";
    $("provider-message").textContent = status.provider_error || "";
    $("connection").textContent = "Índice disponible";
    state.ready = true; updateMode();
  } catch {
    $("connection").textContent = "Sin conexión"; $("connection").classList.add("offline");
    $("provider-message").textContent = "No se pudo conectar con el servidor local. Vuelve a abrir el chat cuando esté disponible.";
  }
}
initialize();
