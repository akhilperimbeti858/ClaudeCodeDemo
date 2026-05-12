"""
Fuzzy string matching layer using rapidfuzz.

Scoring strategy (best of three scorers, then take max across all aliases):
  - token_sort_ratio   – order-invariant token match  (good for name transpositions)
  - token_set_ratio    – handles subsets / extra tokens (good for long legal names)
  - WRatio             – weighted composite             (general fallback)

Returns a score in [0, 100].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rapidfuzz import fuzz, process, utils

from .models import SDNEntity

logger = logging.getLogger(__name__)


@dataclass
class FuzzyCandidate:
    sdn_entity: SDNEntity
    score: float          # [0, 100]
    matched_alias: str    # which SDN name string produced the best score


def _score_pair(query: str, candidate: str) -> float:
    """Compute the best fuzzy score between two strings."""
    q = utils.default_process(query)
    c = utils.default_process(candidate)
    if not q or not c:
        return 0.0
    return max(
        fuzz.token_sort_ratio(q, c),
        fuzz.token_set_ratio(q, c),
        fuzz.WRatio(q, c),
    )


def score_against_sdn(query: str, sdn_entity: SDNEntity) -> tuple[float, str]:
    """
    Return ``(best_score, matched_alias)`` for *query* against all names of
    *sdn_entity* (primary + aliases).
    """
    best_score = 0.0
    best_alias = sdn_entity.name

    for name in sdn_entity.all_names:
        s = _score_pair(query, name)
        if s > best_score:
            best_score = s
            best_alias = name

    return best_score, best_alias


def top_fuzzy_candidates(
    query: str,
    sdn_list: list[SDNEntity],
    threshold: float = 70.0,
    top_k: int = 20,
) -> list[FuzzyCandidate]:
    """
    Run a fast pre-filter with ``process.extract`` on the primary name index,
    then re-score the winners against all aliases and apply *threshold*.

    Parameters
    ----------
    query:      The entity text from Comprehend.
    sdn_list:   Full SDN entity list.
    threshold:  Minimum fuzzy score to keep [0, 100].
    top_k:      Maximum number of candidates to return after filtering.

    Returns a list of FuzzyCandidate sorted descending by score.
    """
    if not sdn_list:
        return []

    # Build a flat name→entity index for the rapid pre-filter
    # Each entry: (all_name_string, sdn_entity)
    name_index: list[tuple[str, SDNEntity]] = []
    for entity in sdn_list:
        for alias in entity.all_names:
            name_index.append((alias, entity))

    name_strings = [n for n, _ in name_index]

    # Extract top-k*3 candidates quickly using WRatio scorer
    raw_hits = process.extract(
        query,
        name_strings,
        scorer=fuzz.WRatio,
        limit=top_k * 3,
        processor=utils.default_process,
        score_cutoff=threshold,
    )

    # Deduplicate by SDN uid — re-score against all aliases for fairness
    seen_uids: set[str] = set()
    candidates: list[FuzzyCandidate] = []

    for _matched_name, _raw_score, idx in raw_hits:
        _, entity = name_index[idx]
        if entity.uid in seen_uids:
            continue
        seen_uids.add(entity.uid)

        best_score, best_alias = score_against_sdn(query, entity)
        if best_score >= threshold:
            candidates.append(
                FuzzyCandidate(
                    sdn_entity=entity,
                    score=best_score,
                    matched_alias=best_alias,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]
