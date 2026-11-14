"use strict";
const $ = id => document.getElementById(id);
const state = {history: [], busy: false, ready: false, sources: [], cited: [], local: false};
const phases = {loading_encoder: "Preparando el modelo local…", retrieving: "Buscando en el manual…", generating: "Redactando con las fuentes…"};

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
    summary.append(labels, element("strong", "", `PI-RADS v2.1 · p. ${source.printed_pages.join(", ")}`));
    summary.append(element("span", "source-meta", `Página física del PDF: ${source.pages.join(", ")} · score ${source.score.toFixed(3)}`));
    summary.append(element("p", "source-preview", source.text.slice(0, 150).replace(/\s+/g, " ") + "…"));
    const link = element("a", "source-pdf", "Abrir página en el PDF ↗");
    link.href = source.pdf_url;
    link.target = "_blank";
    link.rel = "noopener";
    card.append(summary, element("p", "source-text", source.text), link);
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
  $("model").disabled = search || state.busy;
  $("cloud-note").textContent = search
    ? "La búsqueda y los embeddings se ejecutan localmente. Este modo no llama al generador Cloud."
    : state.local ? "Todo se procesa en este equipo: embeddings y respuesta en la RTX. Sin llamadas Cloud ni coste por tokens."
    : "Al enviar, Ollama Cloud recibe la pregunta, el historial reciente y los fragmentos recuperados. La clave permanece en el servidor local.";
  $("send").disabled = state.busy || !state.ready || (!search && !$("model").value);
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || state.busy) return;
  state.busy = true;
  updateMode();
  ["encoder", "collection", "mode", "new-chat"].forEach(id => { $(id).disabled = true; });
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
      collection: $("collection").value, model: $("model").value
    });
    pending.remove();
    if (search) {
      data.answer = "He recuperado cuatro fragmentos del manual. Puedes abrirlos en el panel de fuentes y consultar sus páginas originales.";
      data.cited_sources = [];
    }
    const body = element("div", "answer-body");
    renderAnswer(body, data); turn.append(body);
    const footer = element("div", "answer-footer");
    const seconds = data.metrics.retrieval_seconds + (data.metrics.latency_seconds || 0);
    footer.append(element("span", "", search ? "Búsqueda local · sin generación" : data.metrics.model));
    footer.append(element("span", "", `${seconds.toFixed(1)} s · ${data.sources.length} fuentes`));
    if (data.abstained) footer.append(element("span", "badge", "Evidencia insuficiente"));
    if (data.truncated) footer.append(element("span", "badge", "Salida recortada por el límite de generación"));
    const show = element("button", "", "Ver fuentes ↗");
    show.type = "button";
    show.addEventListener("click", () => showSources(data.sources, data.cited_sources));
    const copy = element("button", "", "Copiar"); copy.type = "button";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(data.answer); copy.textContent = "Copiado"; }
      catch { copy.textContent = "No se pudo copiar"; }
    });
    footer.append(show, copy); turn.append(footer);
    if (!search) state.history.push({role: "user", content: question}, {role: "assistant", content: data.answer});
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
    ["encoder", "collection", "mode", "new-chat"].forEach(id => { $(id).disabled = false; });
    updateMode(); $("question").focus();
  }
}

$("chat-form").addEventListener("submit", sendQuestion);
$("question").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); if (!$("send").disabled) $("chat-form").requestSubmit();
  }
});
$("mode").addEventListener("change", updateMode);
$("model").addEventListener("change", updateMode);
$("toggle-sources").addEventListener("click", () => toggleSources($("sources-panel").hidden));
$("close-sources").addEventListener("click", () => toggleSources(false));
$("new-chat").addEventListener("click", () => {
  state.history = []; state.sources = []; state.cited = [];
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
    state.local = status.provider_kind === "local";
    document.querySelector("label[for=model] .cloud-label").textContent = state.local ? "LOCAL" : "CLOUD";
    $("model").replaceChildren();
    for (const name of status.models) {
      const option = element("option", "", name); option.value = name; $("model").append(option);
    }
    if (status.models.includes(status.model)) $("model").value = status.model;
    else if (status.models.length) $("model").value = "";
    else { const option = element("option", "", "Configura Ollama en .env"); option.value = ""; $("model").append(option); }
    if (!status.generation_ready) $("mode").value = "search";
    $("provider-message").textContent = status.provider_error || "";
    $("connection").textContent = "Índice disponible";
    state.ready = true; updateMode();
  } catch {
    $("connection").textContent = "Sin conexión"; $("connection").classList.add("offline");
    $("provider-message").textContent = "No se pudo conectar con el servidor local. Vuelve a abrir el chat cuando esté disponible.";
  }
}
initialize();
