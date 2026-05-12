"""
Entity flagging: identify non-entities and partial-match fragments.

Two outcome flags:

  NON_ENTITY
    The text extracted by Comprehend is likely noise rather than a real named
    entity.  Causes: too short, numeric-only, low Comprehend confidence, or a
    common English stop-word masquerading as an entity.

  NEEDS_CONTEXT_EXPANSION
    The text is a genuine entity fragment — a partial name that matched against
    a longer SDN entry.  A context-window expansion call (larger surrounding
    text → re-run Comprehend) is needed before a reliable screening decision
    can be made.

    Classic examples:
      "Jihad"       →  "Palestinian Islamic Jihad"
      "Islamic"     →  "Islamic Revolutionary Guard Corps"
      "Bank"        →  "Iran Import and Export Bank"
      "Wagner"      →  "Wagner Private Military Company"

  CLEAN
    No issues detected — the match can be accepted or rejected on score alone.

Heuristics for NEEDS_CONTEXT_EXPANSION
    1. substring_of_match  — query text appears verbatim inside the matched
       SDN name and the query is < 60 % of its length.
    2. token_ratio_gap     — the matched SDN name has ≥ 2 more tokens than
       the query AND combined_score > min_combined.
    3. no_match_fragment   — no matches were found but the query is a single
       short token that could be a name fragment (> 3 chars, not a stop-word).
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from .models import ComprehendEntity, EntityFlag, MatchResult

logger = logging.getLogger(__name__)

# Minimum Comprehend score below which an entity is flagged as NON_ENTITY
_MIN_COMPREHEND_SCORE: float = 0.50

# Ratio of query length to matched-name length below which context expansion
# is triggered (substring_of_match check)
_SUBSTRING_LENGTH_RATIO: float = 0.60

# Minimum combined match score for the token_ratio_gap check to fire.
# Avoids flagging very weak matches as needing expansion (they're just misses).
_MIN_COMBINED_FOR_TOKEN_GAP: float = 0.45

# Single-token queries longer than this that have no match are flagged
_MIN_FRAGMENT_LEN_FOR_EXPANSION: int = 4

# English stop-words and common noise tokens that are never real entity names
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "mr", "ms", "mrs", "dr",
    "inc", "llc", "ltd", "co", "corp", "group", "company",
    # single-letter tokens
    "i", "s", "n",
})

_NUMERIC_RE = re.compile(r"^\d[\d\s\-\.\,]*$")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")


def flag_entity(
    entity: ComprehendEntity,
    matches: list[MatchResult],
    min_combined: float = _MIN_COMBINED_FOR_TOKEN_GAP,
) -> tuple[EntityFlag, str]:
    """
    Evaluate a single Comprehend entity and its best matches.

    Returns ``(EntityFlag, reason_string)``.

    Parameters
    ----------
    entity:       The Comprehend entity being evaluated.
    matches:      Best MatchResult list for this entity (may be empty).
    min_combined: Minimum combined score threshold for token-gap check.
    """
    text = entity.text.strip()
    tokens = text.split()
    lower = text.lower()

    # ------------------------------------------------------------------
    # NON-ENTITY checks
    # ------------------------------------------------------------------

    if len(text) < 3:
        return EntityFlag.NON_ENTITY, "text_too_short"

    if _NUMERIC_RE.match(text):
        return EntityFlag.NON_ENTITY, "numeric_only"

    if _PUNCT_ONLY_RE.match(text):
        return EntityFlag.NON_ENTITY, "punctuation_only"

    if entity.score < _MIN_COMPREHEND_SCORE:
        return EntityFlag.NON_ENTITY, f"low_comprehend_confidence({entity.score:.2f})"

    if len(tokens) == 1 and lower in _STOP_WORDS:
        return EntityFlag.NON_ENTITY, f"common_stop_word({lower!r})"

    # ------------------------------------------------------------------
    # CONTEXT EXPANSION checks
    # ------------------------------------------------------------------

    if matches:
        best = matches[0]
        matched_name = best.layer_score.matched_alias

        # 1. Query is a verbatim substring of the matched SDN name
        if (
            lower in matched_name.lower()
            and len(text) < len(matched_name) * _SUBSTRING_LENGTH_RATIO
        ):
            return (
                EntityFlag.NEEDS_CONTEXT_EXPANSION,
                f"substring_of_match: query {text!r} is a fragment of {matched_name!r}",
            )

        # 2. Matched name has substantially more tokens and score is meaningful
        query_token_count = len(tokens)
        match_token_count = len(matched_name.split())
        if (
            match_token_count >= query_token_count + 2
            and best.combined_score >= min_combined
        ):
            return (
                EntityFlag.NEEDS_CONTEXT_EXPANSION,
                f"token_ratio_gap: {query_token_count} query tokens vs "
                f"{match_token_count} in {matched_name!r}",
            )

    else:
        # 3. No match found — flag single-token fragments for expansion
        if (
            len(tokens) == 1
            and len(text) >= _MIN_FRAGMENT_LEN_FOR_EXPANSION
            and lower not in _STOP_WORDS
        ):
            return (
                EntityFlag.NEEDS_CONTEXT_EXPANSION,
                f"no_match_single_token: {text!r} may be a name fragment",
            )

    return EntityFlag.CLEAN, ""


def apply_flags(
    all_results: list[list[MatchResult]],
    entities: list[ComprehendEntity],
    min_combined: float = _MIN_COMBINED_FOR_TOKEN_GAP,
) -> None:
    """
    In-place: set ``flag`` and ``flag_reason`` on every MatchResult, and
    propagate the entity-level flag to all MatchResults for that entity.

    Parameters
    ----------
    all_results:  Nested list parallel to *entities*.
    entities:     The Comprehend entities that produced the results.
    min_combined: Forwarded to flag_entity().
    """
    for entity, matches in zip(entities, all_results):
        flag, reason = flag_entity(entity, matches, min_combined)
        if flag != EntityFlag.CLEAN:
            logger.info(
                "Flag %s for %r: %s", flag.value, entity.text, reason
            )
        for match in matches:
            match.flag = flag
            match.flag_reason = reason
