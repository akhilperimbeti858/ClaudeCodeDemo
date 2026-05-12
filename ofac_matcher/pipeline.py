"""
OFACMatcher — tiered, silo-aware OFAC entity screening pipeline.

Flow per Comprehend entity
──────────────────────────
1. Comprehend entity type  →  target SDN silos
   (ORGANIZATION → OFAC_ORG + FTO,  PERSON → OFAC_POI, etc.)

2. Fuzzy pre-filter   (rapidfuzz, three tiers, silo-filtered)
   Tier 1: exact match
   Tier 2: Jaro-Winkler  (>= jw_threshold)
   Tier 3: token-sort / token-set / WRatio partial

3. Semantic re-ranking   (FAISS siloed indexes, three tiers)
   FAISS IndexFlatIP (cosine) restricted to silos from step 1
   and further restricted to UIDs surviving step 2.
   Tier 1: cosine >= 0.95
   Tier 2: cosine >= 0.80
   Tier 3: cosine >= semantic_threshold

4. Score fusion
   combined = fuzzy_weight * (fuzzy_score / 100)
            + semantic_weight * cosine_sim

5. Ranking: primary sort key = fuzzy_tier; secondary = -combined_score
   (Exact matches always come before JW before partial, regardless of score)

6. Entity flagging   (entity_flagger)
   NON_ENTITY              — noise / too short / low confidence
   NEEDS_CONTEXT_EXPANSION — probable partial name fragment
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
from .entity_flagger import apply_flags
from .fuzzy_layer import top_fuzzy_candidates
from .models import (
    ComprehendEntity,
    DistanceMetric,
    EntityFlag,
    FuzzyTier,
    LayerScore,
    MatchResult,
    SDNEntity,
    SDNSilo,
    SemanticTier,
)
from .ofac_loader import load_sdn_from_dict, load_sdn_list
from .sdn_classifier import assign_silos, comprehend_to_silos
from .semantic_layer import SemanticIndex

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All tuneable parameters for the matching pipeline."""

    # --- Fuzzy layer ---
    fuzzy_threshold: float = 70.0     # min rapidfuzz score for Tier 3 [0, 100]
    fuzzy_top_k: int = 30             # candidates forwarded to semantic stage
    jw_threshold: float = 0.92        # min Jaro-Winkler similarity for Tier 2 [0, 1]

    # --- Semantic layer ---
    semantic_threshold: float = 0.50  # min cosine similarity [0, 1]
    semantic_top_k: int = 10          # max final results per query entity
    distance_metric: DistanceMetric = DistanceMetric.L2
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 256
    use_tfidf_fallback: bool = False

    # --- Score fusion ---
    fuzzy_weight: float = 0.35        # must sum to 1.0 with semantic_weight
    semantic_weight: float = 0.65

    # --- Comprehend filter ---
    # If non-empty, only these Comprehend entity types are processed.
    entity_type_filter: list[str] = field(default_factory=list)

    # --- Flagging ---
    min_combined_for_flag: float = 0.45

    def __post_init__(self) -> None:
        total = self.fuzzy_weight + self.semantic_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"fuzzy_weight + semantic_weight must equal 1.0, got {total:.4f}"
            )


class OFACMatcher:
    """
    Two-stage, silo-aware OFAC entity matching pipeline.

    Typical usage::

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

        # Assign silos in-place before indexing
        assign_silos(sdn_list)

        self._semantic_index = SemanticIndex(
            model_name=self.config.embedding_model,
            distance_metric=self.config.distance_metric,
            batch_size=self.config.embedding_batch_size,
            use_tfidf_fallback=self.config.use_tfidf_fallback,
        )
        logger.info("Building siloed FAISS indexes for %d SDN entries …", len(sdn_list))
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
        return cls(load_sdn_list(path), config)

    @classmethod
    def from_sdn_records(
        cls,
        records: list[dict],
        config: Optional[PipelineConfig] = None,
    ) -> "OFACMatcher":
        return cls(load_sdn_from_dict(records), config)

    # ------------------------------------------------------------------
    # Internal: match one entity
    # ------------------------------------------------------------------

    def _match_one(
        self,
        entity: ComprehendEntity,
        target_silos: list[SDNSilo],
    ) -> list[MatchResult]:
        cfg = self.config
        query = entity.text

        # Stage 1 — tiered fuzzy search (silo-filtered)
        fuzzy_candidates = top_fuzzy_candidates(
            query=query,
            sdn_list=self._sdn_list,
            target_silos=target_silos,
            threshold=cfg.fuzzy_threshold,
            top_k=cfg.fuzzy_top_k,
            jw_threshold=cfg.jw_threshold,
        )

        if not fuzzy_candidates:
            logger.debug("No fuzzy candidates for %r", query)
            return []

        candidate_uids = {c.sdn_entity.uid for c in fuzzy_candidates}
        fuzzy_by_uid = {c.sdn_entity.uid: c for c in fuzzy_candidates}

        # Stage 2 — FAISS semantic re-ranking (silo-restricted)
        semantic_candidates = self._semantic_index.query(
            query_text=query,
            silos_to_search=target_silos,
            top_k=cfg.semantic_top_k,
            threshold=cfg.semantic_threshold,
            candidate_uids=candidate_uids,
        )

        if not semantic_candidates:
            logger.debug("No semantic candidates for %r", query)
            return []

        # Fuse scores
        results: list[MatchResult] = []
        for sem in semantic_candidates:
            uid = sem.sdn_entity.uid
            fuzz_cand = fuzzy_by_uid.get(uid)

            fuzzy_score = fuzz_cand.score if fuzz_cand else 0.0
            fuzzy_tier = fuzz_cand.tier if fuzz_cand else FuzzyTier.PARTIAL
            fuzzy_alias = fuzz_cand.matched_alias if fuzz_cand else sem.matched_alias
            best_alias = fuzzy_alias if fuzzy_score >= sem.cosine_similarity * 100 else sem.matched_alias

            combined = (
                cfg.fuzzy_weight * (fuzzy_score / 100.0)
                + cfg.semantic_weight * sem.cosine_similarity
            )

            results.append(
                MatchResult(
                    comprehend_entity=entity,
                    sdn_entity=sem.sdn_entity,
                    layer_score=LayerScore(
                        fuzzy_score=fuzzy_score,
                        fuzzy_tier=fuzzy_tier,
                        semantic_score=sem.cosine_similarity,
                        semantic_tier=sem.semantic_tier,
                        distance=sem.distance,
                        matched_alias=best_alias,
                    ),
                    combined_score=combined,
                    silo=sem.silo,
                )
            )

        # Primary sort: fuzzy_tier ASC (Exact < JW < Partial)
        # Secondary sort: combined_score DESC within each tier
        results.sort(key=lambda r: (r.layer_score.fuzzy_tier.value, -r.combined_score))
        for rank, r in enumerate(results, 1):
            r.rank = rank

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        comprehend_input: Union[dict, list[dict], list[ComprehendEntity]],
        input_format: str = "auto",
    ) -> list[list[MatchResult]]:
        """
        Match Comprehend entities against the SDN list.

        Parameters
        ----------
        comprehend_input:
            ``"auto"``      — detect shape automatically
            ``"detect"``    — boto3 detect_entities() response dict
            ``"batch"``     — boto3 batch_detect_entities() response dict
            ``"raw_list"``  — list of raw entity dicts
            ``"entities"``  — list[ComprehendEntity]

        Returns a list parallel to the input entities.  Each element is a
        ``list[MatchResult]`` ordered by (fuzzy_tier ASC, combined_score DESC).
        Flags (CLEAN / NON_ENTITY / NEEDS_CONTEXT_EXPANSION) are set on every
        MatchResult.
        """
        entities = self._coerce_input(comprehend_input, input_format)

        if self.config.entity_type_filter:
            allowed = {t.upper() for t in self.config.entity_type_filter}
            entities = [e for e in entities if e.entity_type.upper() in allowed]

        all_results: list[list[MatchResult]] = []
        for entity in entities:
            target_silos = comprehend_to_silos(entity.entity_type)
            matches = self._match_one(entity, target_silos)
            all_results.append(matches)
            if matches:
                top = matches[0]
                logger.info(
                    "Query %r [%s] → %d match(es) | top: %r silo=%s "
                    "tier=%s combined=%.4f",
                    entity.text, entity.entity_type, len(matches),
                    top.sdn_entity.name, top.silo.value,
                    top.layer_score.fuzzy_tier.name, top.combined_score,
                )
            else:
                logger.info("Query %r [%s] → 0 matches", entity.text, entity.entity_type)

        # Apply entity flags in-place
        apply_flags(all_results, entities, self.config.min_combined_for_flag)
        return all_results

    def match_flat(
        self,
        comprehend_input: Union[dict, list[dict], list[ComprehendEntity]],
        input_format: str = "auto",
        min_combined_score: float = 0.0,
    ) -> list[MatchResult]:
        """
        Like ``match()`` but returns a single flat list of all MatchResults,
        filtered by *min_combined_score* and sorted by
        (fuzzy_tier ASC, combined_score DESC).
        """
        nested = self.match(comprehend_input, input_format)
        flat = [r for matches in nested for r in matches if r.combined_score >= min_combined_score]
        flat.sort(key=lambda r: (r.layer_score.fuzzy_tier.value, -r.combined_score))
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
            fmt == "auto"
            and isinstance(inp, list)
            and inp
            and isinstance(inp[0], ComprehendEntity)
        ):
            return inp  # type: ignore[return-value]

        if fmt == "detect" or (fmt == "auto" and isinstance(inp, dict) and "Entities" in inp):
            return parse_detect_entities_response(inp)  # type: ignore[arg-type]

        if fmt == "batch" or (fmt == "auto" and isinstance(inp, dict) and "ResultList" in inp):
            return parse_batch_response(inp)  # type: ignore[arg-type]

        if fmt == "raw_list" or (
            fmt == "auto" and isinstance(inp, list) and inp and isinstance(inp[0], dict)
        ):
            return from_raw_list(inp)  # type: ignore[arg-type]

        if isinstance(inp, list) and not inp:
            return []

        raise ValueError(
            f"Cannot infer input format for type {type(inp).__name__}. "
            "Set input_format explicitly: 'detect', 'batch', 'raw_list', or 'entities'."
        )
