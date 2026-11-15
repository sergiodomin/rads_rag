"use strict";
const $ = id => document.getElementById(id);
const selected = new Set();
let metadata, activeCase, running = false, timer;
const labels = {retrieval_only: "Sólo recuperación", no_context: "Sin contexto",
  reference_context: "Contexto de referencia", retrieved_context: "RAG generativo", product: "Baseline textual"};
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function message(text = "") { $("message").textContent = text; }
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function checked(name) { return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(x => x.value); }
function details(title, content) {
  const el = node("details"); el.append(node("summary", title), node("pre", typeof content === "string" ? content : JSON.stringify(content, null, 2))); return el;
}
function evidence(parent, sources) {
  for (const source of sources) {
    const box = node("div", undefined, "evidence");
    box.append(node("strong", `${source.id} · ${source.system} · ${source.document_version} · ${source.document_role}`));
    if (source.pdf_url?.startsWith("/documents/")) {
      const link = node("a", "Abrir documento ↗"); link.href = source.pdf_url; link.target = "_blank"; link.rel = "noopener"; box.append(node("br"), link);
    }
    box.append(node("p", source.text), details("Localización y hashes", {...source, text: undefined})); parent.append(box);
  }
}
function renderCases() {
  const split = $("split").value, language = $("language").value;
  const cases = metadata.dataset.cases.filter(c => c.split === split && (language === "all" || c.language === language));
  $("cases").replaceChildren();
  $("split-note").textContent = split === "test" ? "Reserva test para comprobar una configuración ya fijada. No ajustes el sistema con estos resultados." : "Ajusta métodos y revisa referencias en desarrollo.";
  for (const c of cases) {
    const row = node("div", undefined, "case-row"), label = node("label", undefined, "check"), input = node("input");
    input.type = "checkbox"; input.checked = selected.has(c.id);
    input.addEventListener("change", () => { input.checked ? selected.add(c.id) : selected.delete(c.id); count(); });
    label.append(input, node("span", `${c.question} (${c.language} · ${c.type} · ${c.review_status})`));
    const inspect = node("button", "Revisar", "secondary"); inspect.addEventListener("click", () => showReference(c));
    row.append(label, inspect); $("cases").append(row);
  }
  count();
}
function showReference(c) {
  activeCase = c;
  $("reference-panel").open = true;
  $("case-title").textContent = `${c.id} · ${c.split} · ${c.answerable ? "Respondible" : "Control sin evidencia"}`;
  $("case-history").textContent = JSON.stringify(c.history, null, 2);
  $("case-evidence").replaceChildren(); evidence($("case-evidence"), c.evidence.map((e,i) => ({...e.source, id: `R${i+1}`})));
  $("reference-answer").value = c.reference_answer; $("reference-notes").value = c.review_notes;
  $("reference-reviewed").checked = c.review_status === "reviewed";
}
function count() {
  const generation = checked("condition").filter(c => c !== "retrieval_only").length;
  const calls = selected.size * generation * checked("model").length;
  $("selection-count").textContent = `${selected.size} casos seleccionados (incluidos otros filtros)`;
  $("call-count").textContent = `Hasta ${calls} llamadas al LLM. BM25 no carga embeddings. Los lotes comparten exclusión con el chat.`;
  $("start").disabled = running || !selected.size || !checked("condition").length || calls > 12 || (generation > 0 && !checked("model").length);
  $("cancel").disabled = !running;
}
async function refreshRuns() {
  const runs = await api("/api/lab/runs");
  for (const id of ["run-a", "run-b"]) {
    const previous = $(id).value;
    $(id).replaceChildren(new Option("Ninguna", ""));
    for (const run of runs) $(id).add(new Option(`${run.created_at.slice(0,19)} · ${run.label} · ${run.status} · ${run.completed}/${run.total}`, run.run_id));
    $(id).value = previous || (id === "run-a" && runs.length ? runs[0].run_id : "");
  }
}
function reviewForm(record, runId) {
  const panel = node("details"); panel.append(node("summary", "Revisar salida bruta por afirmación"));
  const form = node("form"), fields = {};
  const values = {
    faithfulness: ["unreviewed", "supported", "partial", "contradicted", "not_applicable"],
    correctness: ["unreviewed", "correct", "partial", "incorrect", "not_applicable"],
    citation_support: ["unreviewed", "supported", "partial", "unsupported", "not_applicable"],
    abstention: ["unreviewed", "appropriate", "excessive", "missing", "not_applicable"]};
  const names = {faithfulness: "Fidelidad al contexto", correctness: "Corrección frente a referencia", citation_support: "Apoyo de las citas", abstention: "Abstención"};
  for (const [key, options] of Object.entries(values)) {
    const label = node("label", names[key]), select = node("select");
    options.forEach(v => select.add(new Option(v, v))); select.value = record.human_review?.[key] || "unreviewed";
    fields[key] = select; label.append(select); form.append(label);
  }
  const notesLabel = node("label", "Notas"), notes = node("textarea"); notes.maxLength = 4000; notes.value = record.human_review?.notes || ""; notesLabel.append(notes); form.append(notesLabel);
  const claims = node("div");
  function claimRow(value = {}) {
    const box = node("div", undefined, "claim"), textLabel = node("label", "Afirmación"), text = node("textarea");
    text.maxLength = 2000; text.value = value.text || ""; textLabel.append(text);
    const judgmentLabel = node("label", "Valoración"), judgment = node("select");
    ["supported", "contradicted", "not_in_evidence"].forEach(v => judgment.add(new Option(v,v))); judgment.value = value.judgment || "not_in_evidence"; judgmentLabel.append(judgment);
    const sourceLabel = node("label", "IDs de evidencia separados por coma (S1, R1…)"), sources = node("input"); sources.value = (value.source_ids || []).join(", "); sourceLabel.append(sources);
    const remove = node("button", "Quitar", "secondary"); remove.type = "button"; remove.onclick = () => box.remove();
    box.append(textLabel, judgmentLabel, sourceLabel, remove);
    box.readClaim = () => ({text: text.value, judgment: judgment.value, source_ids: sources.value.split(",").map(x => x.trim()).filter(Boolean)});
    claims.append(box);
  }
  (record.human_review?.claims || []).forEach(claimRow);
  const add = node("button", "Añadir afirmación", "secondary"); add.type = "button"; add.onclick = () => { if (claims.children.length < 30) claimRow(); };
  const save = node("button", "Guardar revisión"); save.type = "submit";
  const feedback = node("p"); feedback.setAttribute("role", "status");
  form.append(claims, add, node("br"), save, feedback);
  form.onsubmit = async event => {
    event.preventDefault(); save.disabled = true;
    try { await api("/api/lab/result-review", {run_id: runId, record_id: record.id,
      ...Object.fromEntries(Object.entries(fields).map(([key, field]) => [key, field.value])),
      notes: notes.value, claims: [...claims.children].map(c => c.readClaim())}); feedback.textContent = "Revisión guardada."; }
    catch (error) { feedback.textContent = error.message; }
    finally { save.disabled = false; }
  };
  panel.append(form); return panel;
}
function rankingView(trace) {
  const panel = node("details"); panel.append(node("summary", "Candidatos · antes, después y contexto final"));
  const table = node("table"), head = node("thead"), tr = node("tr");
  ["Inicial", "Rerank", "Score reranker", "Fuente final", "Pasaje"].forEach(x => tr.append(node("th", x))); head.append(tr); table.append(head);
  const body = node("tbody");
  for (const candidate of trace.candidates || []) {
    const row = node("tr");
    [candidate.rank, candidate.rerank_rank ?? "—", candidate.rerank_score == null ? "—" : candidate.rerank_score.toFixed(4), candidate.selected_rank ? `S${candidate.selected_rank}` : "—"].forEach(x => row.append(node("td", String(x))));
    const info = node("td"); info.append(node("strong", `${candidate.document_id} · ${candidate.chunk_id}`), node("p", candidate.preview));
    if (candidate.pair?.document_truncated) info.append(node("p", `Reranker: documento recortado a ${candidate.pair.scored_document_tokens} tokens.`, "muted"));
    row.append(info); body.append(row);
  }
  table.append(body); const scroll = node("div", undefined, "ranking-scroll"); scroll.append(table); panel.append(scroll);
  const pool = node("details"); pool.append(node("summary", `Texto y procedencia de los candidatos (${trace.candidate_sources?.length || 0})`)); evidence(pool, trace.candidate_sources || []); panel.append(pool);
  return panel;
}
function renderRecord(record, runId) {
  const box = node("article", undefined, "record");
  box.append(node("h3", `${record.case_id} · ${labels[record.condition]}`), node("p", record.question));
  box.append(node("p", `${record.model || "Sin LLM"} · ${record.elapsed_seconds.toFixed(2)} s · ${record.status} · referencias: ${record.reference_status}`, "muted"));
  if (record.error) box.append(node("blockquote", record.error));
  if (record.raw_answer !== null) {
    box.append(node("strong", "Salida bruta"), node("blockquote", record.raw_answer));
    if (record.answer !== record.raw_answer) box.append(node("strong", "Tras controles"), node("blockquote", record.answer));
    box.append(node("p", `Modo: ${record.answer_mode || "Generación libre"} · LLM ejecutado: ${record.generation_executed ? "sí" : "no"} · abstención: ${record.abstention_reason || "no"}`, "muted"));
  }
  box.append(details("Métricas automáticas · sin juicio clínico", record.automatic), details("Tiempos, tokens y presupuesto", record.metrics || {}));
  if (record.trace.retrieval) box.append(rankingView(record.trace.retrieval));
  const context = node("details"); context.append(node("summary", `Contexto utilizado (${record.context_sources.length})`)); evidence(context, record.context_sources); box.append(context);
  const refs = node("details"); refs.append(node("summary", "Referencia congelada de esta ejecución"), node("pre", record.reference_answer)); evidence(refs, record.reference_sources); box.append(refs);
  box.append(details("Traza · consultas, ranking y mensajes", record.trace));
  if (record.raw_answer !== null) box.append(reviewForm(record, runId));
  return box;
}
async function renderRuns() {
  const ids = [$("run-a").value, $("run-b").value];
  const runs = await Promise.all(ids.map(id => id ? api(`/api/lab/runs/${id}`) : null));
  const columns = runs.map((run, i) => {
    const col = node("div", undefined, "run-column");
    col.append(node("h3", `Ejecución ${i === 0 ? "A" : "B"}`));
    if (!run) { col.append(node("p", "Selecciona una ejecución.", "muted")); return col; }
    col.append(details("Resumen · referencias y tamaños de muestra", run.manifest.summary));
    const exportLink = node("a", "Descargar resultados JSON"); exportLink.href = `/api/lab/runs/${ids[i]}`; exportLink.download = `lab-${ids[i]}.json`; col.append(exportLink);
    for (const record of run.records) col.append(renderRecord(record, ids[i]));
    return col;
  });
  $("results").replaceChildren(...columns);
}
async function poll() {
  clearTimeout(timer);
  try {
    const state = await api("/api/lab/state"); running = state?.status === "running";
    $("progress").textContent = state ? `${state.status} · ${state.completed}/${state.total} · ${state.current_task?.case_id || ""} ${state.current_task?.condition || ""}` : "Sin ejecuciones activas.";
    count();
    if (running) timer = setTimeout(poll, 2000);
    else { await refreshRuns(); if (state?.run_id && !$("run-a").value) $("run-a").value = state.run_id; await renderRuns(); }
  } catch (error) { message(error.message); running = false; count(); }
}
async function initialize() {
  try {
    metadata = await api("/api/lab");
    if (!metadata.dataset) throw new Error("Prepara el dataset: uv run python scripts/prepare_lab_dataset.py");
    $("dataset-count").textContent = `${metadata.dataset.cases.length} casos · rev. ${metadata.dataset.revision}`;
    metadata.encoders.forEach(v => $("encoder").add(new Option(v,v))); $("encoder").value = "qwen3-0.6b";
    (metadata.rerankers || []).forEach(v => $("reranker").add(new Option(`${v.id} · ${v.max_pair_tokens} tokens/par`,v.id)));
    for (const generator of metadata.generators) {
      const label = node("label", undefined, "check"), input = node("input"); input.type = "checkbox"; input.name = "model"; input.value = generator.model; input.checked = true; input.onchange = count; label.append(input, node("span", generator.model)); $("models").append(label);
    }
    selected.add(metadata.dataset.cases.find(c => c.split === "dev" && c.language === "es").id);
    renderCases(); await refreshRuns(); await poll();
  } catch (error) { message(error.message); $("start").disabled = true; }
}
$("split").onchange = renderCases; $("language").onchange = renderCases;
$("select-visible").onclick = () => { metadata.dataset.cases.filter(c => c.split === $("split").value && ($("language").value === "all" || c.language === $("language").value)).forEach(c => selected.add(c.id)); renderCases(); };
$("clear-selection").onclick = () => { selected.clear(); renderCases(); };
document.querySelectorAll('input[name="condition"]').forEach(el => el.onchange = count);
$("start").onclick = async () => {
  message(); running = true; count();
  try {
    const state = await api("/api/lab/start", {case_ids: [...selected], conditions: checked("condition"), models: checked("model"), encoder: $("encoder").value, retriever: $("retriever").value, collection: $("collection").value, temperature: Number($("temperature").value), num_predict: Number($("output-tokens").value), reranker: $("reranker").value, candidate_count: Number($("candidate-count").value), context_policy: $("context-policy").value});
    await refreshRuns(); $("run-a").value = state.run_id; await poll();
  } catch (error) { running = false; count(); message(error.message); }
};
$("cancel").onclick = async () => { try { await api("/api/lab/cancel", {}); message("Detención solicitada. La tarea actual debe terminar."); } catch (error) { message(error.message); } };
$("save-reference").onclick = async () => {
  if (!activeCase) return;
  try {
    await api("/api/lab/reference-review", {case_id: activeCase.id, reference_answer: $("reference-answer").value, notes: $("reference-notes").value, reviewed: $("reference-reviewed").checked});
    const id = activeCase.id; metadata = await api("/api/lab"); renderCases(); showReference(metadata.dataset.cases.find(c => c.id === id)); message("Referencia guardada; las ejecuciones anteriores conservan su snapshot.");
  } catch (error) { message(error.message); }
};
for (const id of ["run-a", "run-b"]) $(id).onchange = () => renderRuns().catch(error => message(error.message));
initialize();
