"""
OFACMatcher — two-stage entity matching pipeline.

Stage 1 – Fuzzy pre-filter (rapidfuzz)
    Fast string matching narrows thousands of SDN entries down to a small
    candidate set.  Scores are token-order-invariant and handle abbreviations.

Stage 2 – Semantic re-ranking (sentence-transformers + L1/L2)
    A transformer-based encoder embeds both the query and the candidates.
    Cosine similarity re-ranks and filters the fuzzy candidates.

Final score
    combined = fuzzy_weight * (fuzzy_score / 100) + semantic_weight * cosine_sim
    where fuzzy_weight + semantic_weight = 1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .comprehend_parser import (
    from_raw_list,
    parse_async_job_output,
    parse_batch_response,
    parse_detect_entities_response,
)
from .fuzzy_layer import top_fuzzy_candidates
from .models import ComprehendEntity, DistanceMetric, LayerScore, MatchResult, SDNEntity
from .ofac_loader import load_sdn_from_dict, load_sdn_list
from .semantic_layer import SemanticIndex

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Tuneable knobs for the matching pipeline."""

    # --- Fuzzy layer ---
    fuzzy_threshold: float = 70.0    # min rapidfuzz score [0, 100] to pass stage 1
    fuzzy_top_k: int = 30            # how many candidates stage 1 forwards to stage 2

    # --- Semantic layer ---
    semantic_threshold: float = 0.50  # min cosine similarity [0, 1] to keep
    semantic_top_k: int = 10          # max final results per query entity
    distance_metric: DistanceMetric = DistanceMetric.L2
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 256
    # Set True to skip sentence-transformers and use TF-IDF char n-gram embeddings.
    # Useful for offline environments or fast prototyping.
    use_tfidf_fallback: bool = False

    # --- Score fusion ---
    fuzzy_weight: float = 0.35        # weight for normalised fuzzy score [0, 1]
    semantic_weight: float = 0.65     # weight for cosine similarity [0, 1]

    # --- Comprehend filter ---
    # If non-empty, only Comprehend entities of these types are matched.
    # e.g. ["PERSON", "ORGANIZATION"]
    entity_type_filter: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        total = self.fuzzy_weight + self.semantic_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"fuzzy_weight + semantic_weight must equal 1.0, got {total:.4f}"
            )


class OFACMatcher:
    """
    Two-stage OFAC entity matching pipeline.

    Usage::

        matcher = OFACMatcher.from_sdn_file("sdn.xml")
        results = matcher.match(comprehend_response)

        for entity_results in results:
            for match in entity_results:
                print(match)
    """

    def __init__(
        self,
        sdn_list: list[SDNEntity],
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._sdn_list = sdn_list

        self._semantic_index = SemanticIndex(
            model_name=self.config.embedding_model,
            distance_metric=self.config.distance_metric,
            batch_size=self.config.embedding_batch_size,
            use_tfidf_fallback=self.config.use_tfidf_fallback,
        )
        logger.info("Building semantic index for %d SDN entries …", len(sdn_list))
        self._semantic_index.build_index(sdn_list)
        logger.info("OFACMatcher ready.")

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_sdn_file(
        cls,
        path: Union[str, Path],
        config: Optional[PipelineConfig] = None,
    ) -> "OFACMatcher":
        """Load from the official OFAC XML or CSV file on disk."""
        sdn_list = load_sdn_list(path)
        return cls(sdn_list, config)

    @classmethod
    def from_sdn_records(
        cls,
        records: list[dict],
        config: Optional[PipelineConfig] = None,
    ) -> "OFACMatcher":
        """
        Build from a plain list of dicts (e.g. loaded from a DB or test fixture).
        Each dict must have at least: ``uid``, ``name``.
        Optional: ``entity_type``, ``aliases``, ``programs``.
        """
        sdn_list = load_sdn_from_dict(records)
        return cls(sdn_list, config)

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def _match_one(self, entity: ComprehendEntity) -> list[MatchResult]:
        """Run the two-stage pipeline for a single Comprehend entity."""
        cfg = self.config
        query = entity.text

        # Stage 1 — Fuzzy pre-filter
        fuzzy_candidates = top_fuzzy_candidates(
            query=query,
            sdn_list=self._sdn_list,
            threshold=cfg.fuzzy_threshold,
            top_k=cfg.fuzzy_top_k,
        )

        if not fuzzy_candidates:
            logger.debug("No fuzzy candidates for %r (threshold=%.1f)", query, cfg.fuzzy_threshold)
            return []

        candidate_uids = {c.sdn_entity.uid for c in fuzzy_candidates}
        fuzzy_by_uid = {c.sdn_entity.uid: c for c in fuzzy_candidates}

        # Stage 2 — Semantic re-ranking within the candidate set
        semantic_candidates = self._semantic_index.query(
            query_text=query,
            top_k=cfg.semantic_top_k,
            threshold=cfg.semantic_threshold,
            candidate_uids=candidate_uids,
        )

        # Fuse scores
        results: list[MatchResult] = []
        for sem in semantic_candidates:
            uid = sem.sdn_entity.uid
            fuzz_cand = fuzzy_by_uid.get(uid)

            fuzzy_score = fuzz_cand.score if fuzz_cand else 0.0
            fuzzy_alias = fuzz_cand.matched_alias if fuzz_cand else sem.matched_alias

            # Use the alias with the higher individual score as the display alias
            best_alias = (
                fuzzy_alias if fuzzy_score >= sem.cosine_similarity * 100 else sem.matched_alias
            )

            normalised_fuzzy = fuzzy_score / 100.0
            combined = (
                cfg.fuzzy_weight * normalised_fuzzy
                + cfg.semantic_weight * sem.cosine_similarity
            )

            results.append(
                MatchResult(
                    comprehend_entity=entity,
                    sdn_entity=sem.sdn_entity,
                    layer_score=LayerScore(
                        fuzzy_score=fuzzy_score,
                        semantic_score=sem.cosine_similarity,
                        distance=sem.distance,
                        matched_alias=best_alias,
                    ),
                    combined_score=combined,
                )
            )

        results.sort(key=lambda r: r.combined_score, reverse=True)
        for rank, r in enumerate(results, 1):
            r.rank = rank

        return results

    def match(
        self,
        comprehend_input: Union[
            dict,                    # raw detect_entities() or batch_detect_entities() response
            list[dict],              # list of raw entity dicts
            list[ComprehendEntity],  # already-parsed entities
        ],
        input_format: str = "auto",
    ) -> list[list[MatchResult]]:
        """
        Match Comprehend entities against the SDN list.

        Parameters
        ----------
        comprehend_input:
            Accepts multiple shapes — see *input_format*.
        input_format:
            ``"auto"``        – detect shape automatically (default)
            ``"detect"``      – boto3 detect_entities() response dict
            ``"batch"``       – boto3 batch_detect_entities() response dict
            ``"raw_list"``    – list of raw entity dicts
            ``"entities"``    – list[ComprehendEntity] (already parsed)

        Returns
        -------
        A list parallel to the input entities.  Each element is a
        ``list[MatchResult]`` sorted by combined_score descending.
        """
        entities = self._coerce_input(comprehend_input, input_format)

        # Apply entity type filter
        if self.config.entity_type_filter:
            allowed = {t.upper() for t in self.config.entity_type_filter}
            entities = [e for e in entities if e.entity_type.upper() in allowed]

        all_results: list[list[MatchResult]] = []
        for entity in entities:
            matches = self._match_one(entity)
            all_results.append(matches)
            logger.info(
                "Query %r → %d match(es) (top: %s, score=%.4f)",
                entity.text,
                len(matches),
                matches[0].sdn_entity.name if matches else "—",
                matches[0].combined_score if matches else 0.0,
            )

        return all_results

    def match_flat(
        self,
        comprehend_input: Union[dict, list[dict], list[ComprehendEntity]],
        input_format: str = "auto",
        min_combined_score: float = 0.0,
    ) -> list[MatchResult]:
        """
        Like ``match()`` but returns a single flat list of all MatchResults
        across all query entities, filtered by *min_combined_score*.
        """
        nested = self.match(comprehend_input, input_format)
        flat = [r for matches in nested for r in matches]
        flat = [r for r in flat if r.combined_score >= min_combined_score]
        flat.sort(key=lambda r: r.combined_score, reverse=True)
        return flat

    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------

    def _coerce_input(
        self,
        inp: Union[dict, list[dict], list[ComprehendEntity]],
        fmt: str,
    ) -> list[ComprehendEntity]:
        if fmt == "entities" or (
            fmt == "auto" and inp and isinstance(inp, list) and isinstance(inp[0], ComprehendEntity)
        ):
            return inp  # type: ignore[return-value]

        if fmt == "detect" or (fmt == "auto" and isinstance(inp, dict) and "Entities" in inp):
            return parse_detect_entities_response(inp)  # type: ignore[arg-type]

        if fmt == "batch" or (fmt == "auto" and isinstance(inp, dict) and "ResultList" in inp):
            return parse_batch_response(inp)  # type: ignore[arg-type]

        if fmt == "raw_list" or (fmt == "auto" and isinstance(inp, list) and inp and isinstance(inp[0], dict)):
            return from_raw_list(inp)  # type: ignore[arg-type]

        if isinstance(inp, list) and not inp:
            return []

        raise ValueError(
            f"Cannot infer input format for type {type(inp).__name__}. "
            "Set input_format explicitly: 'detect', 'batch', 'raw_list', or 'entities'."
        )
