"""
Three-tier fuzzy matching layer using rapidfuzz.

Tier 1 – EXACT
    Normalised exact string equality after lowercasing, stripping punctuation,
    and collapsing whitespace.  Score = 100.0.

Tier 2 – JARO_WINKLER
    Jaro-Winkler similarity >= jw_threshold (default 0.92).  Strong on
    short-name transpositions and typographic variants ("AL-QAIDA" / "AL QAEDA").
    Score = jaro_winkler * 100.

Tier 3 – PARTIAL
    Best of: token_sort_ratio, token_set_ratio, WRatio.  Handles word-order
    differences, abbreviations, and long legal name fragments.

Ordering guarantee
    Tier 1 results always precede Tier 2, which always precede Tier 3.
    Within a tier results are sorted descending by score.

Silo pre-filter
    Only SDN entities whose silo is in *target_silos* are considered.
    This enforces the rule "if Comprehend said ORGANIZATION, only search
    OFAC_ORG and FTO — never OFAC_POI".
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

from rapidfuzz import fuzz, process, utils
from rapidfuzz.distance import JaroWinkler

from .models import FuzzyTier, SDNEntity, SDNSilo

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]")


def _norm(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


# ---------------------------------------------------------------------------
# Per-candidate dataclass
# ---------------------------------------------------------------------------

@dataclass
class FuzzyCandidate:
    sdn_entity: SDNEntity
    score: float          # [0, 100]
    matched_alias: str    # which SDN name string produced this score
    tier: FuzzyTier


# ---------------------------------------------------------------------------
# Tier-specific scorers
# ---------------------------------------------------------------------------

def _exact_score(query: str, candidate: str) -> bool:
    return _norm(query) == _norm(candidate)


def _jw_score(query: str, candidate: str) -> float:
    return JaroWinkler.similarity(query.lower(), candidate.lower())


def _partial_score(query: str, candidate: str) -> float:
    q = utils.default_process(query)
    c = utils.default_process(candidate)
    if not q or not c:
        return 0.0
    return max(
        fuzz.token_sort_ratio(q, c),
        fuzz.token_set_ratio(q, c),
        fuzz.WRatio(q, c),
    )


# ---------------------------------------------------------------------------
# Per-entity best score across all aliases (returns best alias too)
# ---------------------------------------------------------------------------

def _best_exact(query: str, entity: SDNEntity) -> tuple[bool, str]:
    for name in entity.all_names:
        if _exact_score(query, name):
            return True, name
    return False, ""


def _best_jw(query: str, entity: SDNEntity, threshold: float) -> tuple[float, str]:
    best, best_name = 0.0, entity.name
    for name in entity.all_names:
        s = _jw_score(query, name)
        if s > best:
            best, best_name = s, name
    if best >= threshold:
        return best, best_name
    return 0.0, ""


def _best_partial(query: str, entity: SDNEntity) -> tuple[float, str]:
    best, best_name = 0.0, entity.name
    for name in entity.all_names:
        s = _partial_score(query, name)
        if s > best:
            best, best_name = s, name
    return best, best_name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def top_fuzzy_candidates(
    query: str,
    sdn_list: list[SDNEntity],
    target_silos: list[SDNSilo],
    threshold: float = 70.0,
    top_k: int = 30,
    jw_threshold: float = 0.92,
) -> list[FuzzyCandidate]:
    """
    Run the three-tier fuzzy search and return up to *top_k* candidates,
    ordered by tier then descending score.

    Parameters
    ----------
    query:          Entity text from Comprehend.
    sdn_list:       Full SDN list (already silo-classified).
    target_silos:   Silos to search — entities outside these silos are skipped.
    threshold:      Minimum rapidfuzz score for Tier 3 [0, 100].
    top_k:          Maximum candidates to return (applied after merging tiers).
    jw_threshold:   Minimum Jaro-Winkler similarity for Tier 2 [0, 1].
    """
    # Filter to target silos
    pool = [e for e in sdn_list if e.silo in target_silos]
    if not pool:
        logger.debug("No SDN entries in silos %s for query %r", target_silos, query)
        return []

    tier1: list[FuzzyCandidate] = []
    tier2: list[FuzzyCandidate] = []

    # --- Tier 1: exact ---
    exact_uids: set[str] = set()
    for entity in pool:
        matched, alias = _best_exact(query, entity)
        if matched:
            tier1.append(FuzzyCandidate(entity, 100.0, alias, FuzzyTier.EXACT))
            exact_uids.add(entity.uid)

    # --- Tier 2: Jaro-Winkler (skip already-exact-matched entries) ---
    jw_uids: set[str] = set(exact_uids)
    for entity in pool:
        if entity.uid in exact_uids:
            continue
        jw, alias = _best_jw(query, entity, jw_threshold)
        if jw > 0.0:
            tier2.append(FuzzyCandidate(entity, jw * 100.0, alias, FuzzyTier.JARO_WINKLER))
            jw_uids.add(entity.uid)

    # --- Tier 3: partial (skip Tier 1 + 2) ---
    # Use rapidfuzz process.extract for speed on the full name corpus
    name_index: list[tuple[str, SDNEntity]] = []
    for entity in pool:
        if entity.uid in jw_uids:
            continue
        for alias in entity.all_names:
            name_index.append((alias, entity))

    tier3: list[FuzzyCandidate] = []
    if name_index:
        name_strings = [n for n, _ in name_index]
        raw_hits = process.extract(
            query,
            name_strings,
            scorer=fuzz.WRatio,
            limit=top_k * 4,
            processor=utils.default_process,
            score_cutoff=threshold,
        )
        seen_t3: set[str] = set()
        for _matched_name, _raw_score, idx in raw_hits:
            _, entity = name_index[idx]
            if entity.uid in seen_t3 or entity.uid in jw_uids:
                continue
            seen_t3.add(entity.uid)
            best, alias = _best_partial(query, entity)
            if best >= threshold:
                tier3.append(FuzzyCandidate(entity, best, alias, FuzzyTier.PARTIAL))

    tier1.sort(key=lambda c: -c.score)
    tier2.sort(key=lambda c: -c.score)
    tier3.sort(key=lambda c: -c.score)

    all_candidates = tier1 + tier2 + tier3
    logger.debug(
        "Fuzzy: query=%r  T1=%d T2=%d T3=%d  silos=%s",
        query, len(tier1), len(tier2), len(tier3),
        [s.value for s in target_silos],
    )
    return all_candidates[:top_k]
