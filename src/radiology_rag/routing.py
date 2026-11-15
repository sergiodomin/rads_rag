"""Una entrada de chat; planificación explícita de evidencia sin otro modelo cargado."""

import re

from radiology_rag.chat import validate_request
from radiology_rag.fusion import FusedRAG
from radiology_rag.retrieval.hybrid import fold, normalize_question, query_intent

VISUAL_CUES = re.compile(
    r"\b(?:color(?:es)?|colou?rs?|flechas?|arrows?|diagramas?|diagrams?|figuras?|figures?|"
    r"graficos?|charts?|paneles|panels?|leyendas?|legends?|mapa(?:s)? sectorial(?:es)?|"
    r"sector maps?|recortes?|crops?)\b"
)
IMAGE_CUES = re.compile(r"\b(?:imagen(?:es)?|images?|dibujo|drawing)\b")
IMAGE_ACTION = re.compile(
    r"\b(?:muestra|muestrame|ver|observa|representa|describe|shows?|see|look)\b"
)
FOLLOWUP = re.compile(r"^(?:y\b|and\b|what about\b|que hay\b)")
TEXT_CUES = re.compile(
    r"\b(?:seguimiento|follow.?up|manejo|management|criterios?|criteria|define|definicion|"
    r"definition|tratamiento|treatment|riesgo|risk|probabilidad|probability)\b"
)
COMBINED_CUES = re.compile(r"\b(?:explica|explain|significa|means?|notas?|notes?|texto|text)\b")


class UnifiedChat:
    def __init__(self, engine, visual):
        self.engine, self.visual = engine, visual

    def plan(self, payload):
        question, _, _, _ = validate_request(payload)
        normalized = normalize_question(question)[0]
        query = fold(normalized).strip(" ¿?¡!")
        mode = payload.get("modality", "auto")
        previous = payload.get("visual_context", "")
        if (
            not isinstance(mode, str)
            or mode not in {"auto", "text", "visual", "combined"}
            or not isinstance(previous, str)
            or len(previous) > 100
        ):
            raise ValueError("Plan de evidencia o contexto visual no válidos.")
        system = payload.get("system", "PI-RADS")
        if not isinstance(system, str) or system not in {"all", "PI-RADS", "BI-RADS"}:
            raise ValueError("Sistema documental no disponible.")
        visual_intent = bool(VISUAL_CUES.search(query))
        visual_intent |= bool(IMAGE_CUES.search(query) and IMAGE_ACTION.search(query))
        visual_intent |= bool(
            re.search(r"\b(?:tabla|table)\b", query) and IMAGE_ACTION.search(query)
        )
        # Color Doppler is a textual imaging topic unless a graphic is explicitly requested.
        if "doppler" in query and not re.search(r"\b(?:figura|figure|diagrama|diagram)\b", query):
            visual_intent = False
        followup = bool(
            previous
            and FOLLOWUP.search(query)
            and len(query) <= 160
            and not TEXT_CUES.search(query)
        )
        named, _ = query_intent(normalized)
        if followup and named:
            assets = self.visual.metadata().get("assets", [])
            old = next((a for a in assets if a["id"] == previous), None)
            if not old or named != {old["system"]}:
                followup = False
        if mode == "text" or (mode == "auto" and not visual_intent and not followup):
            return {"modality": "text", "reason": "text_and_structured_tables", "context": ""}
        identifier = payload.get("visual_asset", "") or (previous if followup else "")
        if not isinstance(identifier, str) or len(identifier) > 100:
            raise ValueError("Región visual no válida.")
        try:
            asset, _ = self.visual.select(normalized, system, identifier)
            figure = re.search(r"\b(?:figura|figure)\s*(?:numero\s*)?(\d+)\b", query)
            if (
                not identifier
                and figure
                and not asset.get("xml_element_id", "").endswith(f"-f{int(figure[1]):03d}")
            ):
                raise ValueError("Esa figura no está registrada para lectura visual.")
            if (
                not identifier
                and re.search(r"\b(?:tabla|table)\b", query)
                and asset["kind"] != "table"
            ):
                raise ValueError("Esa tabla todavía no tiene un recorte registrado.")
            if (
                not identifier
                and re.search(r"\b(?:diagrama|diagram|flecha|arrow)", query)
                and asset["kind"] != "flow_diagram"
            ):
                raise ValueError("No hay un diagrama de flujo registrado que corresponda.")
        except ValueError as error:
            return {
                "modality": "clarification",
                "reason": "visual_region_unavailable",
                "detail": str(error),
                "context": "",
            }
        combined = mode == "combined" or bool(COMBINED_CUES.search(query))
        return {
            "modality": "combined" if combined else "image",
            "reason": "visual_followup" if followup else "visual_evidence_required",
            "asset_id": asset["id"],
            "context": asset["id"],
        }

    def respond(self, payload):
        plan = self.plan(payload)
        if (
            plan["modality"] != "clarification"
            and (self.engine.root / "data/catalog/vision_generator.v1.json").exists()
        ):
            return FusedRAG(self.engine, self.visual).respond(payload, plan)
        if plan["modality"] == "text":
            result = self.engine.respond(payload)
        elif plan["modality"] in {"image", "combined"}:
            result = self.visual.respond(
                {**payload, "visual_asset": plan["asset_id"], "visual_input": plan["modality"]}
            )
        else:
            result = {
                "answer": "Para responder necesito una región visual disponible. " + plan["detail"],
                "sources": [],
                "cited_sources": [],
                "abstained": True,
                "abstention_reason": "visual_region_unavailable",
                "truncated": False,
                "answer_mode": "clarification",
                "metrics": {
                    "model": "Selección de evidencia",
                    "retrieval_seconds": 0,
                    "latency_seconds": 0,
                },
            }
        result["routing"] = {"strategy": "rules_and_registered_regions_v1", **plan}
        return result
