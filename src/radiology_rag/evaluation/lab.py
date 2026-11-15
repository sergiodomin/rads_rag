"""Experimentos locales secuenciales, snapshots y revisión humana independiente."""

import json
import math
import re
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version

from tokenizers import Tokenizer

from radiology_rag.chat import NO_EVIDENCE, BusyError, check_citations, grounded_messages
from radiology_rag.evaluation.lab_dataset import (
    DATASET,
    atomic_json,
    evidence_matches,
    load_dataset,
    oracle_sources,
    validate_dataset,
)
from radiology_rag.generation.context import context_budget
from radiology_rag.generation.ollama import OllamaCloudClient
from radiology_rag.ingestion.catalog import sha256_file
from radiology_rag.ingestion.parsing import fingerprint
from radiology_rag.paths import within
from radiology_rag.retrieval.hybrid import normalize_question, overview_comparison, query_intent
from radiology_rag.retrieval.reranking import catalog as reranker_catalog
from radiology_rag.retrieval.reranking import model_spec as reranker_spec

CONDITIONS = {"retrieval_only", "no_context", "reference_context", "retrieved_context", "product"}
NO_CONTEXT_PROMPT = """Responde en el idioma de la pregunta a partir de tu conocimiento previo.
No se han proporcionado documentos. No inventes referencias a fuentes ni IDs de cita.
Si no sabes la respuesta, dilo. Esto es un control experimental sin contexto documental,
no una respuesta clínica validada. No clasifiques casos de pacientes."""
REVIEW_VALUES = {
    "faithfulness": {"unreviewed", "supported", "partial", "contradicted", "not_applicable"},
    "correctness": {"unreviewed", "correct", "partial", "incorrect", "not_applicable"},
    "citation_support": {"unreviewed", "supported", "partial", "unsupported", "not_applicable"},
    "abstention": {"unreviewed", "appropriate", "excessive", "missing", "not_applicable"},
}


def now():
    return datetime.now(UTC).isoformat()


def retrieval_assessment(case, sources):
    evidence = case["evidence"]
    if not evidence:
        return {
            "evidence_recall": None,
            "mrr": None,
            "novel_evidence_ndcg": None,
            "all_evidence": None,
            "reference_status": case["review_status"],
            "note": "Sin evidencia positiva anotada; no se calcula recall.",
        }
    covered = set()
    gains = []
    for source in sources:
        hits = {e["id"] for e in evidence if evidence_matches(source, e)}
        gains.append(float(bool(hits - covered)))
        covered.update(hits)
    first = next((i + 1 for i, gain in enumerate(gains) if gain), None)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(evidence), len(sources))))
    return {
        "evidence_recall": len(covered) / len(evidence),
        "all_evidence": len(covered) == len(evidence),
        "mrr": 1 / first if first else 0,
        "novel_evidence_ndcg": dcg / ideal if ideal else 0,
        "covered_evidence_ids": sorted(covered),
        "reference_status": case["review_status"],
        "note": "Ganancia binaria por fuente que añade evidencia nueva; no es exactitud clínica.",
    }


def summarize(records):
    arms = {}
    for record in records:
        key = (
            record["condition"],
            record.get("model"),
            record["retriever"],
            record["encoder"],
            record.get("reranker", "none"),
            record.get("context_policy", "parent"),
        )
        arm = arms.setdefault(
            key,
            {
                "condition": key[0],
                "model": key[1],
                "retriever": key[2],
                "encoder": key[3],
                "reranker": key[4],
                "context_policy": key[5],
                "records": 0,
                "errors": 0,
                "evidence_values": [],
                "seconds": [],
                "reviewed_references": 0,
                "generation_calls": 0,
                "abstentions": 0,
                "truncated": 0,
                "human_reviews": 0,
                "human_judgments": {},
                "breakdowns": {},
            },
        )
        arm["records"] += 1
        arm["errors"] += record["status"] == "error"
        arm["generation_calls"] += bool(record.get("generation_executed"))
        arm["abstentions"] += bool(record.get("abstained"))
        arm["truncated"] += bool(record.get("truncated"))
        review = record.get("human_review")
        if review:
            arm["human_reviews"] += 1
            for field in REVIEW_VALUES:
                judgments = arm["human_judgments"].setdefault(field, {})
                judgment = review.get(field, "unreviewed")
                judgments[judgment] = judgments.get(judgment, 0) + 1
        value = record.get("automatic", {}).get("retrieval", {}).get("evidence_recall")
        if value is not None:
            arm["evidence_values"].append(value)
        for field in ("type", "language", "split"):
            bucket = (
                arm["breakdowns"]
                .setdefault(field, {})
                .setdefault(record[field], {"records": 0, "errors": 0, "retrieval_values": []})
            )
            bucket["records"] += 1
            bucket["errors"] += record["status"] == "error"
            if value is not None:
                bucket["retrieval_values"].append(value)
        if record["reference_status"] == "reviewed":
            arm["reviewed_references"] += 1
        arm["seconds"].append(record.get("elapsed_seconds", 0))
    result = []
    for arm in arms.values():
        values, seconds = arm.pop("evidence_values"), arm.pop("seconds")
        arm["mean_evidence_recall"] = sum(values) / len(values) if values else None
        arm["retrieval_sample_size"] = len(values)
        arm["mean_elapsed_seconds"] = sum(seconds) / len(seconds) if seconds else None
        arm["semantic_accuracy"] = None
        for buckets in arm["breakdowns"].values():
            for bucket in buckets.values():
                measures = bucket.pop("retrieval_values")
                bucket["retrieval_sample_size"] = len(measures)
                bucket["mean_evidence_recall"] = sum(measures) / len(measures) if measures else None
        arm["metric_status"] = (
            "reviewed_references"
            if arm["reviewed_references"] == arm["records"]
            else "draft_references_not_validated"
        )
        result.append(arm)
    return result


class EvaluationLab:
    def __init__(self, engine):
        self.engine, self.root = engine, engine.root
        self.folder = self.root / "output/lab"
        self.session_id = uuid.uuid4().hex
        self._mutex, self._cancel = threading.Lock(), threading.Event()
        self._thread, self._current = None, None
        self._tokenizers = {}

    def generator_profiles(self):
        path = self.root / "data/catalog/lab_generators.v1.json"
        if not path.exists():
            return {}
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry["schema_version"] != 1:
            raise ValueError("Registro de generadores no compatible.")
        result = {}
        for entry in registry["generators"]:
            profile_path = within(self.root, entry["profile"])
            if not profile_path.is_relative_to((self.root / "data/catalog").resolve()):
                raise ValueError("Perfil de generador fuera del catálogo.")
            spec = json.loads(profile_path.read_text(encoding="utf-8"))
            if entry["budget_format"] != "qwen3_chatml":
                raise ValueError("Plantilla de presupuesto todavía no soportada.")
            result[spec["model"]] = {
                **spec,
                "profile": entry["profile"],
                "profile_sha256": sha256_file(profile_path),
                "budget_format": entry["budget_format"],
            }
        return result

    def metadata(self):
        data = load_dataset(self.root) if (self.root / DATASET).exists() else None
        return {
            "dataset": data,
            "dataset_sha256": fingerprint(data) if data else None,
            "generators": [
                {
                    "model": name,
                    "digest": p.get("ollama_digest"),
                    "budget_format": p["budget_format"],
                }
                for name, p in self.generator_profiles().items()
            ],
            "encoders": list(self.engine.specs),
            "conditions": sorted(CONDITIONS),
            "retrievers": ["hybrid", "dense", "bm25"],
            "rerankers": [
                {"id": s["id"], "revision": s["revision"], "max_pair_tokens": s["max_pair_tokens"]}
                for s in reranker_catalog(self.root)
            ],
            "max_generation_calls": 12,
            "local_only": True,
            "current": self.state(),
            "runs": self.list_runs(),
        }

    def state(self):
        with self._mutex:
            return deepcopy(self._current)

    def _run_path(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("ID de ejecución no válido.")
        return self.folder / "runs" / run_id

    def read_run(self, run_id):
        folder = self._run_path(run_id)
        if not (folder / "run.json").exists():
            raise ValueError("Ejecución no encontrada.")
        manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        records = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((folder / "records").glob("*.json"))
        ]
        for record in records:
            review = folder / "reviews" / f"{record['id']}.json"
            record["human_review"] = (
                json.loads(review.read_text(encoding="utf-8")) if review.exists() else None
            )
            record["semantic_review"] = (
                "human_review_recorded" if record["human_review"] else "pending_human_review"
            )
        if manifest["status"] == "running" and manifest.get("session_id") != self.session_id:
            manifest["status"] = "interrupted"
        snapshot = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
        manifest["summary"] = summarize(records)
        manifest["summary_basis"] = "records_with_current_human_reviews"
        return {"manifest": manifest, "records": records, "snapshot": snapshot}

    def list_runs(self):
        paths = sorted(
            (self.folder / "runs").glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:30]
        result = []
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["status"] == "running" and data.get("session_id") != self.session_id:
                data["status"] = "interrupted"
            result.append(
                {
                    k: data.get(k)
                    for k in (
                        "run_id",
                        "label",
                        "status",
                        "created_at",
                        "completed",
                        "total",
                        "summary",
                    )
                }
            )
        return result

    def _validate_config(self, config, dataset):
        if not isinstance(config, dict):
            raise ValueError("Configuración no válida.")
        ids, conditions = config.get("case_ids", []), config.get("conditions", [])
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 64
            or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Selecciona entre 1 y 64 casos, sin duplicados.")
        if (
            not isinstance(conditions, list)
            or not conditions
            or any(not isinstance(c, str) or c not in CONDITIONS for c in conditions)
            or len(set(conditions)) != len(conditions)
        ):
            raise ValueError("Condiciones de experimento no válidas.")
        cases = {c["id"]: c for c in dataset["cases"]}
        if not set(ids) <= cases.keys():
            raise ValueError("Caso desconocido.")
        models = config.get("models", [])
        profiles = self.generator_profiles()
        generation = any(c != "retrieval_only" for c in conditions)
        if (
            not isinstance(models, list)
            or len(models) > 2
            or any(not isinstance(m, str) or m not in profiles for m in models)
            or len(models) != len(set(models))
            or (generation and not models)
        ):
            raise ValueError(
                "Selecciona hasta dos generadores registrados con tokenizer preparado."
            )
        calls = len(ids) * sum(c != "retrieval_only" for c in conditions) * len(models)
        if calls > 12:
            raise ValueError(
                "Máximo 12 llamadas de generación por lote; reduce casos o condiciones."
            )
        encoder = config.get("encoder", "qwen3-0.6b")
        retriever = config.get("retriever", "hybrid")
        collection = config.get("collection", "structural_512")
        if not isinstance(encoder, str) or encoder not in self.engine.specs:
            raise ValueError("Encoder desconocido.")
        if (
            not isinstance(retriever, str)
            or not isinstance(collection, str)
            or retriever not in {"hybrid", "dense", "bm25"}
            or collection
            not in {
                "structural_512",
                "fixed_512_15",
            }
        ):
            raise ValueError("Recuperador o colección no válidos.")
        for identifier in ids:
            for doc_id, names in self.engine.document_collections.items():
                system = self.engine.documents[doc_id].get("system", "PI-RADS")
                if cases[identifier]["system"] in {"all", system} and collection not in names:
                    raise ValueError("BI-RADS necesita colección estructural.")
        temperature = config.get("temperature", 0.0)
        reranker = config.get("reranker", "none")
        candidate_count = config.get("candidate_count", 20)
        context_policy = config.get("context_policy", "parent")
        if (
            not isinstance(reranker, str)
            or type(candidate_count) is not int
            or candidate_count not in {20, 40}
            or not isinstance(context_policy, str)
            or context_policy not in {"parent", "chunk"}
        ):
            raise ValueError("Reranking, candidatos o contexto no válidos.")
        if reranker != "none":
            reranker_spec(self.root, reranker)
        output = config.get("num_predict", 512)
        if (
            not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 1
            or type(output) is not int
            or not 64 <= output <= 512
        ):
            raise ValueError("Temperatura 0..1 y salida 64..512 tokens.")
        return {
            "case_ids": ids,
            "conditions": conditions,
            "models": models,
            "encoder": encoder,
            "retriever": retriever,
            "reranker": reranker,
            "candidate_count": candidate_count,
            "context_policy": context_policy,
            "collection": collection,
            "temperature": temperature,
            "num_predict": output,
            "num_ctx": self.engine.settings.num_ctx,
        }

    def _native_metadata(self):
        if not self.engine.settings.is_local:
            raise ValueError("El laboratorio sólo ejecuta modelos locales; no llama a Cloud.")
        with OllamaCloudClient(replace(self.engine.settings, max_retries=0)) as client:
            tags, _ = client._request("GET", "/api/tags")
            runtime, _ = client._request("GET", "/api/version")
        return {
            "runtime": runtime.get("version"),
            "models": {
                m["name"]: {"digest": m.get("digest"), "size": m.get("size")}
                for m in tags.get("models", [])
                if not m.get("remote_host") and not m["name"].endswith((":cloud", "-cloud"))
            },
        }

    def start(self, config):
        dataset = load_dataset(self.root)
        config = self._validate_config(config, dataset)
        if not self.engine.settings.is_local:
            raise ValueError("Laboratorio configurado sólo para ejecución local.")
        profiles = self.generator_profiles()
        native = (
            self._native_metadata()
            if any(c != "retrieval_only" for c in config["conditions"])
            else {"runtime": None, "models": {}}
        )
        generation = any(c != "retrieval_only" for c in config["conditions"])
        for model in config["models"] if generation else []:
            if model not in native["models"]:
                raise ValueError("Modelo no instalado; el laboratorio no descarga modelos.")
            expected = profiles[model].get("ollama_digest")
            if expected and native["models"].get(model, {}).get("digest") != expected:
                raise ValueError("El digest instalado no coincide con el perfil de " + model)
        tasks = [
            (case, condition, model)
            for case in config["case_ids"]
            for condition in config["conditions"]
            for model in ([None] if condition == "retrieval_only" else config["models"])
        ]
        with self._mutex:
            if self._thread is not None and self._thread.is_alive():
                raise BusyError("Ya hay una evaluación en curso.")
            if not self.engine._lock.acquire(blocking=False):
                raise BusyError("El chat está ocupado. Espera a que termine.")
            try:
                for doc in self.engine.documents.values():
                    if (
                        doc.get("expected_sha256")
                        and sha256_file(within(self.root, doc["local_path"]))
                        != doc["expected_sha256"]
                    ):
                        raise ValueError("Ha cambiado un documento; reinicia y revisa su índice.")
                    if (
                        doc.get("repository_xml_sha256")
                        and sha256_file(within(self.root, doc["local_path"]).with_suffix(".xml"))
                        != doc["repository_xml_sha256"]
                    ):
                        raise ValueError("Ha cambiado un XML; reinicia y revisa su índice.")
                for case in dataset["cases"]:
                    for item in case["evidence"]:
                        source = item["source"]
                        doc = self.engine.documents.get(source["document_id"])
                        if not doc or source["source_sha256"] != doc.get("expected_sha256"):
                            raise ValueError("Una referencia pertenece a otro corpus.")
                        if source.get("xml_sha256") != doc.get("repository_xml_sha256"):
                            raise ValueError("Una referencia pertenece a otro XML.")
                run_id = uuid.uuid4().hex
                folder = self._run_path(run_id)
                snapshot = {
                    "dataset": dataset,
                    "config": config,
                    "dataset_sha256": fingerprint(dataset),
                    "index_runs": self.engine.run_id,
                    "reranker_profile": (
                        reranker_spec(self.root, config["reranker"], verify=True)
                        if config["reranker"] != "none"
                        else None
                    ),
                    "encoder_profiles": self.engine.specs,
                    "document_catalog": self.engine.documents,
                    "cache_policy": "No cache clearing; Ollama keep_alive=0; sequential models",
                    "generators": {m: profiles[m] for m in config["models"]},
                    "native": native,
                    "environment": {p: version(p) for p in ("numpy", "httpx", "tokenizers")},
                    "code_sha256": {
                        p: sha256_file(self.root / p)
                        for p in (
                            "src/radiology_rag/evaluation/lab.py",
                            "src/radiology_rag/evaluation/lab_dataset.py",
                            "src/radiology_rag/chat.py",
                            "src/radiology_rag/retrieval/hybrid.py",
                            "src/radiology_rag/retrieval/reranking.py",
                            "src/radiology_rag/generation/scope.py",
                            "src/radiology_rag/generation/terminology.py",
                            "src/radiology_rag/generation/context.py",
                            "src/radiology_rag/generation/ollama.py",
                            "src/radiology_rag/ingestion/jats.py",
                            "data/catalog/system_profiles.v1.json",
                        )
                    },
                }
                atomic_json(folder / "snapshot.json", snapshot)
                self._current = {
                    "run_id": run_id,
                    "label": (
                        config["retriever"]
                        + " / "
                        + config["encoder"]
                        + " / "
                        + config["reranker"]
                        + " / "
                        + config["context_policy"]
                    ),
                    "status": "running",
                    "created_at": now(),
                    "completed": 0,
                    "total": len(tasks),
                    "current_task": None,
                    "session_id": self.session_id,
                    "summary": [],
                }
                self._cancel.clear()
                self.engine._phase = "lab_evaluating"
                atomic_json(folder / "run.json", self._current)
                self._thread = threading.Thread(
                    target=self._work, args=(snapshot, tasks, folder), daemon=True
                )
                self._thread.start()
                return deepcopy(self._current)
            except Exception:
                self.engine._phase = "ready"
                self.engine._lock.release()
                raise

    def cancel(self):
        self._cancel.set()
        return {"cancel_requested": True, "note": "La llamada actual termina antes de detenerse."}

    def _budget(self, profile, settings, messages):
        key = profile["profile_sha256"]
        if key not in self._tokenizers:
            spec = profile["tokenizer"]
            folder = within(self.root, spec["path"])
            for item in spec["files"]:
                if sha256_file(within(folder, item["path"])) != item["sha256"]:
                    raise ValueError("Tokenizer de generador modificado.")
            tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self._tokenizers[key] = tokenizer
        return context_budget(
            messages,
            self._tokenizers[key],
            num_ctx=settings.num_ctx,
            num_predict=settings.num_predict,
        )

    def _evaluate(self, case, condition, model, config, profiles):
        started = time.perf_counter()
        reference = oracle_sources(case)
        record = {
            "id": uuid.uuid4().hex,
            "case_id": case["id"],
            "group_id": case["group_id"],
            "split": case["split"],
            "language": case["language"],
            "type": case["type"],
            "question": case["question"],
            "history": case.get("history", []),
            "reference_status": case["review_status"],
            "answerable": case["answerable"],
            "reference_answer": case["reference_answer"],
            "condition": condition,
            "model": model,
            "encoder": config["encoder"],
            "retriever": config["retriever"],
            "reranker": config.get("reranker", "none"),
            "context_policy": config.get("context_policy", "parent"),
            "status": "complete",
            "reference_sources": [{**s, "id": f"R{i}"} for i, s in enumerate(reference, 1)],
            "sources": [],
            "context_sources": [],
            "trace": {},
            "automatic": {},
            "raw_answer": None,
            "answer": None,
            "generation_executed": False,
            "generation_attempted": False,
            "semantic_review": "pending_human_review",
        }
        try:
            sources, metrics = [], {}
            if condition in {"retrieval_only", "retrieved_context", "product"}:
                sources, metrics = self.engine._retrieve(
                    case["question"],
                    case.get("history", []),
                    config["encoder"],
                    config["collection"],
                    case["system"],
                    method=config["retriever"],
                    include_trace=True,
                    reranker=config.get("reranker", "none"),
                    candidate_count=config.get("candidate_count", 20),
                    context_policy=config.get("context_policy", "parent"),
                )
                record["sources"] = sources
                record["trace"]["retrieval"] = metrics.pop("retrieval_trace")
                record["automatic"]["retrieval"] = retrieval_assessment(case, sources)
                trace = record["trace"]["retrieval"]
                if "baseline_sources" in trace:
                    record["automatic"]["retrieval_before_reranking"] = retrieval_assessment(
                        case, trace["baseline_sources"]
                    )
                    record["automatic"]["retrieval_candidate_pool"] = retrieval_assessment(
                        case, trace["candidate_sources"]
                    )
                metrics["context_characters"] = sum(len(s["text"]) for s in sources)
            record["metrics"] = metrics
            context = reference if condition == "reference_context" else sources
            record["context_sources"] = context
            if condition == "retrieval_only":
                return record
            if condition == "product":
                scoped = self.engine._scope_answer(case["question"], sources, metrics)
                if scoped is not None:
                    record.update(
                        {
                            k: scoped[k]
                            for k in (
                                "answer",
                                "abstained",
                                "abstention_reason",
                                "cited_sources",
                                "truncated",
                            )
                        }
                    )
                    record["raw_answer"] = record["answer"]
                    record["answer_mode"] = scoped["answer_mode"]
                    record["metrics"] = scoped["metrics"]
                    return record
            self.engine.release_encoder()
            settings = replace(
                self.engine.settings,
                model=model,
                temperature=config["temperature"],
                num_predict=config["num_predict"],
                max_retries=0,
            )
            normalized, _ = normalize_question(case["question"])
            systems, categories = query_intent(normalized)
            history = [] if len(systems) > 1 else case.get("history", [])
            messages = (
                [
                    {"role": "system", "content": NO_CONTEXT_PROMPT},
                    *history,
                    {"role": "user", "content": case["question"]},
                ]
                if condition == "no_context"
                else grounded_messages(
                    case["question"],
                    history,
                    context,
                    normalized_question=normalized,
                    overview=len(systems) > 1 and overview_comparison(normalized, categories),
                    comparison=len(systems) > 1,
                )
            )
            record["trace"]["messages"] = messages
            record["metrics"].update(self._budget(profiles[model], settings, messages))
            self.engine._phase = "lab_generating"
            record["generation_attempted"] = True
            with OllamaCloudClient(settings) as client:
                generated = client.chat(messages)
            record["generation_executed"] = True
            record["raw_answer"] = generated.content
            record["metrics"].update(generated.metrics())
            record["truncated"] = generated.done_reason in {"length", "max_tokens"}
            cited = set(re.findall(r"\[(S\d+)\]", generated.content))
            allowed = {s["id"] for s in context}
            record["automatic"].update(
                {
                    "citation_ids_valid": (bool(cited) and cited <= allowed)
                    if condition != "no_context"
                    else None,
                    "citation_semantic_support": None,
                    "citations_without_context": bool(cited) if condition == "no_context" else None,
                }
            )
            if condition == "no_context":
                record.update(
                    answer=generated.content,
                    cited_sources=[],
                    abstained=None,
                    abstention_reason="not_applicable_prior_knowledge_control",
                )
            else:
                answer, citations, abstained, reason = check_citations(generated.content, context)
                if len(systems) > 1 and not abstained:
                    cited_systems = {s["system"] for s in context if s["id"] in citations}
                    if not systems <= cited_systems:
                        answer, citations, abstained, reason = (
                            NO_EVIDENCE,
                            [],
                            True,
                            "missing_comparison_sources",
                        )
                record.update(
                    answer=answer,
                    cited_sources=citations,
                    abstained=abstained,
                    abstention_reason=reason,
                )
        except Exception as error:
            record.update(status="error", error=str(error)[:700])
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
        return record

    def _work(self, snapshot, tasks, folder):
        records = []
        cases = {c["id"]: c for c in snapshot["dataset"]["cases"]}
        try:
            for number, (case_id, condition, model) in enumerate(tasks, 1):
                if self._cancel.is_set():
                    break
                with self._mutex:
                    self._current["current_task"] = {
                        "case_id": case_id,
                        "condition": condition,
                        "model": model,
                    }
                self.engine._phase = "lab_evaluating"
                record = self._evaluate(
                    cases[case_id], condition, model, snapshot["config"], snapshot["generators"]
                )
                records.append(record)
                atomic_json(folder / "records" / f"{number:04d}.json", record)
                with self._mutex:
                    self._current["completed"] = number
                    self._current["summary"] = summarize(records)
                    atomic_json(folder / "run.json", self._current)
            with self._mutex:
                self._current["status"] = (
                    "cancelled"
                    if self._cancel.is_set()
                    else (
                        "completed_with_errors"
                        if any(r["status"] == "error" for r in records)
                        else "complete"
                    )
                )
        except Exception as error:
            with self._mutex:
                self._current.update(status="failed", error=str(error)[:500])
        finally:
            try:
                self.engine.release_encoder()
            finally:
                self.engine._phase = "ready"
                self.engine._lock.release()
                with self._mutex:
                    self._current["finished_at"] = now()
                    self._current["current_task"] = None
                    atomic_json(folder / "run.json", self._current)

    def review_case(self, payload):
        with self._mutex:
            if self._thread and self._thread.is_alive():
                raise BusyError("Espera a terminar el lote antes de editar referencias.")
            data = load_dataset(self.root)
            case = next((c for c in data["cases"] if c["id"] == payload.get("case_id")), None)
            if case is None:
                raise ValueError("Caso desconocido.")
            answer, notes = payload.get("reference_answer"), payload.get("notes", "")
            reviewed = payload.get("reviewed", False)
            if (
                not isinstance(answer, str)
                or len(answer) > 20000
                or not isinstance(notes, str)
                or len(notes) > 4000
                or type(reviewed) is not bool
            ):
                raise ValueError("Revisión de referencia no válida.")
            atomic_json(self.root / f"data/evaluations/lab_history/{fingerprint(data)}.json", data)
            case.update(
                reference_answer=answer,
                review_notes=notes,
                review_status="reviewed" if reviewed else "draft",
                reviewed_at=now(),
            )
            data["revision"] += 1
            validate_dataset(data)
            atomic_json(self.root / DATASET, data)
            return {"saved": True, "revision": data["revision"]}

    def review_record(self, payload):
        run_id, record_id = payload.get("run_id"), payload.get("record_id")
        run = self.read_run(run_id)
        record = next((r for r in run["records"] if r["id"] == record_id), None)
        if record is None:
            raise ValueError("Resultado desconocido.")
        review = {key: payload.get(key, "unreviewed") for key in REVIEW_VALUES}
        if any(
            not isinstance(value, str) or value not in REVIEW_VALUES[key]
            for key, value in review.items()
        ):
            raise ValueError("Valoración no válida.")
        notes = payload.get("notes", "")
        claims = payload.get("claims", [])
        if (
            not isinstance(notes, str)
            or len(notes) > 4000
            or not isinstance(claims, list)
            or len(claims) > 30
        ):
            raise ValueError("Revisión demasiado larga.")
        allowed = {s["id"] for s in record["context_sources"] + record["reference_sources"]}
        for claim in claims:
            if (
                not isinstance(claim, dict)
                or not isinstance(claim.get("text"), str)
                or not 1 <= len(claim["text"]) <= 2000
                or not isinstance(claim.get("judgment"), str)
                or claim.get("judgment") not in {"supported", "contradicted", "not_in_evidence"}
                or not isinstance(claim.get("source_ids"), list)
                or any(not isinstance(s, str) or s not in allowed for s in claim["source_ids"])
            ):
                raise ValueError("Afirmación o referencia de revisión no válida.")
        review.update(notes=notes, claims=claims, variant="raw_answer", reviewed_at=now())
        with self._mutex:
            atomic_json(self._run_path(run_id) / "reviews" / f"{record_id}.json", review)
        return {"saved": True}
