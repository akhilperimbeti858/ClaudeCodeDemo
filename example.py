"""
example.py — demonstrates the tiered, silo-aware OFAC matching pipeline.

Features shown
  ✓ Silo routing: ORGANIZATION queries never search OFAC_POI and vice-versa
  ✓ Fuzzy tiers:  Exact → Jaro-Winkler → Partial
  ✓ Semantic tiers via FAISS siloed indexes (cosine: STRONG / GOOD / PARTIAL)
  ✓ Entity flags: CLEAN / NON_ENTITY / NEEDS_CONTEXT_EXPANSION
  ✓ Partial-name fragment detection (e.g. "Jihad" → "Palestinian Islamic Jihad")

Run:
    python example.py

No file download is needed — a stub SDN list is embedded below.
Swap OFACMatcher.from_sdn_records(...) for OFACMatcher.from_sdn_file("sdn.xml")
to use the real OFAC list.
"""

import logging

from ofac_matcher import (
    DistanceMetric,
    EntityFlag,
    OFACMatcher,
    PipelineConfig,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Stub SDN list (replace with load_sdn_list("sdn.xml") in production)
# ---------------------------------------------------------------------------

SDN_STUB = [
    # --- OFAC_ORG (organisations) ---
    {
        "uid": "1001",
        "name": "IRAN IMPORT AND EXPORT BANK",
        "entity_type": "Entity",
        "aliases": ["IIMB", "Iran Import Bank"],
        "programs": ["IRAN"],
    },
    {
        "uid": "1004",
        "name": "MAHAN AIR",
        "entity_type": "Entity",
        "aliases": ["MAHAN AIRLINES", "MAHAN AIRWAYS"],
        "programs": ["IRAN", "SDGT"],
    },
    {
        "uid": "1005",
        "name": "SBERBANK OF RUSSIA",
        "entity_type": "Entity",
        "aliases": ["SBERBANK", "JOINT STOCK COMPANY SBERBANK OF RUSSIA"],
        "programs": ["UKRAINE-EO13685"],
    },
    # --- FTO (foreign terrorist organisations — SDGT programme) ---
    {
        "uid": "1002",
        "name": "AL-QAIDA",
        "entity_type": "Entity",
        "aliases": ["AL QAEDA", "AL QA'IDA", "AL-QAEDA"],
        "programs": ["SDGT"],
    },
    {
        "uid": "1006",
        "name": "HEZBOLLAH",
        "entity_type": "Entity",
        "aliases": ["HIZBALLAH", "HIZBOLLAH", "ISLAMIC JIHAD ORGANIZATION"],
        "programs": ["SDGT"],
    },
    {
        "uid": "1007",
        "name": "WAGNER GROUP",
        "entity_type": "Entity",
        "aliases": ["PMC WAGNER", "WAGNER PRIVATE MILITARY COMPANY"],
        "programs": ["RUSSIA-EO14024"],
    },
    {
        "uid": "1009",
        "name": "PALESTINIAN ISLAMIC JIHAD",
        "entity_type": "Entity",
        "aliases": ["PIJ", "HARAKAT AL-JIHAD AL-ISLAMI AL-FILASTINI"],
        "programs": ["SDGT"],
    },
    {
        "uid": "1010",
        "name": "ISLAMIC REVOLUTIONARY GUARD CORPS",
        "entity_type": "Entity",
        "aliases": ["IRGC", "IRANIAN REVOLUTIONARY GUARD CORPS", "SEPAH"],
        "programs": ["IRAN", "SDGT"],
    },
    # --- OFAC_POI (individuals) ---
    {
        "uid": "1003",
        "name": "KIM JONG UN",
        "entity_type": "Individual",
        "aliases": ["KIM JONGUN", "KIM, JONG UN"],
        "programs": ["DPRK"],
    },
    {
        "uid": "1008",
        "name": "VLADIMIR PUTIN",
        "entity_type": "Individual",
        "aliases": ["PUTIN, VLADIMIR VLADIMIROVICH"],
        "programs": ["RUSSIA-EO14024"],
    },
]

# ---------------------------------------------------------------------------
# Comprehend detect_entities() response — includes deliberate edge cases
# ---------------------------------------------------------------------------

COMPREHEND_RESPONSE = {
    "Entities": [
        # --- Normal matches ---
        {
            "Text": "Iran Import Bank",
            "Type": "ORGANIZATION",
            "Score": 0.9987,
            "BeginOffset": 0, "EndOffset": 16,
        },
        {
            "Text": "Al-Qaeda",
            "Type": "ORGANIZATION",
            "Score": 0.9721,
            "BeginOffset": 20, "EndOffset": 28,
        },
        {
            "Text": "Kim Jong-un",
            "Type": "PERSON",
            "Score": 0.9944,
            "BeginOffset": 32, "EndOffset": 43,
        },
        {
            "Text": "Mahan Airways",
            "Type": "ORGANIZATION",
            "Score": 0.8831,
            "BeginOffset": 47, "EndOffset": 60,
        },
        # --- Silo cross-check: PERSON query should NOT match organisations ---
        {
            "Text": "Wagner Group",
            "Type": "ORGANIZATION",
            "Score": 0.9200,
            "BeginOffset": 64, "EndOffset": 76,
        },
        # --- Context expansion: partial name fragment ---
        {
            "Text": "Jihad",
            "Type": "ORGANIZATION",
            "Score": 0.7800,
            "BeginOffset": 80, "EndOffset": 85,
        },
        {
            "Text": "Islamic",
            "Type": "ORGANIZATION",
            "Score": 0.7100,
            "BeginOffset": 89, "EndOffset": 96,
        },
        # --- Non-entity: too short ---
        {
            "Text": "Al",
            "Type": "PERSON",
            "Score": 0.6500,
            "BeginOffset": 100, "EndOffset": 102,
        },
        # --- Non-entity: low confidence ---
        {
            "Text": "Corporation",
            "Type": "ORGANIZATION",
            "Score": 0.3100,
            "BeginOffset": 105, "EndOffset": 116,
        },
        # --- True non-match (legitimate entity, not on SDN list) ---
        {
            "Text": "ACME Corporation",
            "Type": "ORGANIZATION",
            "Score": 0.7200,
            "BeginOffset": 120, "EndOffset": 136,
        },
    ],
    "ResponseMetadata": {"HTTPStatusCode": 200},
}


# ---------------------------------------------------------------------------
# Helper: print a results block
# ---------------------------------------------------------------------------

_FLAG_ICONS = {
    EntityFlag.CLEAN: "✓",
    EntityFlag.NON_ENTITY: "✗",
    EntityFlag.NEEDS_CONTEXT_EXPANSION: "⚠",
}


def _print_entity_block(entity_matches: list) -> None:
    if not entity_matches:
        return
    e = entity_matches[0].comprehend_entity
    flag = entity_matches[0].flag
    icon = _FLAG_ICONS.get(flag, "?")
    print(
        f"\n{icon} Query: {e.text!r}  [{e.entity_type}  conf={e.score:.0%}]"
        f"  flag={flag.value}"
    )
    if entity_matches[0].flag_reason:
        print(f"    reason: {entity_matches[0].flag_reason}")

    if not entity_matches:
        return
    print(
        f"  {'Rk':<3} {'FzTier':<12} {'SemTier':<9} {'Silo':<10} "
        f"{'Combined':>8} {'Fuzzy':>7} {'Cosine':>7} {'Dist':>7}  SDN Name"
    )
    print("  " + "─" * 100)
    for r in entity_matches:
        print(
            f"  {r.rank:<3} "
            f"{r.layer_score.fuzzy_tier.name:<12} "
            f"{r.layer_score.semantic_tier.name:<9} "
            f"{r.silo.value:<10} "
            f"{r.combined_score:>8.4f} "
            f"{r.layer_score.fuzzy_score:>7.1f} "
            f"{r.layer_score.semantic_score:>7.4f} "
            f"{r.layer_score.distance:>7.4f}  "
            f"{r.sdn_entity.name!r}"
        )


def run_demo() -> None:
    config = PipelineConfig(
        fuzzy_threshold=60.0,
        fuzzy_top_k=25,
        jw_threshold=0.88,
        semantic_threshold=0.35,
        semantic_top_k=5,
        distance_metric=DistanceMetric.L2,    # swap to L1 for Manhattan
        embedding_model="all-MiniLM-L6-v2",
        fuzzy_weight=0.35,
        semantic_weight=0.65,
        use_tfidf_fallback=True,              # offline mode; set False when model available
        min_combined_for_flag=0.40,
    )

    matcher = OFACMatcher.from_sdn_records(SDN_STUB, config=config)

    # ------------------------------------------------------------------
    # Full nested results
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TIERED RESULTS  (Exact > Jaro-Winkler > Partial, then by score)")
    print("Legend:  ✓ CLEAN  ✗ NON_ENTITY  ⚠ NEEDS_CONTEXT_EXPANSION")
    print("=" * 75)

    nested = matcher.match(COMPREHEND_RESPONSE, input_format="detect")
    for entity_matches in nested:
        if entity_matches:
            _print_entity_block(entity_matches)
        else:
            e_list = [
                r.comprehend_entity
                for block in nested
                for r in block
            ]

    # Print zero-match entities too
    from ofac_matcher import from_raw_list
    all_entities = matcher._coerce_input(COMPREHEND_RESPONSE, "detect")
    for entity, matches in zip(all_entities, nested):
        if not matches:
            from ofac_matcher.entity_flagger import flag_entity
            flag, reason = flag_entity(entity, [])
            icon = _FLAG_ICONS.get(flag, "?")
            print(
                f"\n{icon} Query: {entity.text!r}  [{entity.entity_type}  "
                f"conf={entity.score:.0%}]  flag={flag.value}  — no match"
            )
            if reason:
                print(f"    reason: {reason}")

    # ------------------------------------------------------------------
    # Flat results — only CLEAN matches above threshold
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("CLEAN FLAT RESULTS  (combined_score >= 0.55,  flag=CLEAN)")
    print("=" * 75)

    flat = matcher.match_flat(COMPREHEND_RESPONSE, min_combined_score=0.55)
    clean_flat = [r for r in flat if r.flag == EntityFlag.CLEAN]

    print(f"\n  {'Query':<22} {'SDN Name':<38} {'Silo':<10} {'FzTier':<12} {'Combined':>8}")
    print("  " + "─" * 98)
    for r in clean_flat:
        print(
            f"  {r.comprehend_entity.text:<22} "
            f"{r.sdn_entity.name:<38} "
            f"{r.silo.value:<10} "
            f"{r.layer_score.fuzzy_tier.name:<12} "
            f"{r.combined_score:>8.4f}"
        )

    # ------------------------------------------------------------------
    # Entities needing context expansion
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("CONTEXT EXPANSION QUEUE  (fragments that need wider context)")
    print("=" * 75)

    all_flat = matcher.match_flat(COMPREHEND_RESPONSE, min_combined_score=0.0)
    context_entities = {
        r.comprehend_entity.text
        for r in all_flat
        if r.flag == EntityFlag.NEEDS_CONTEXT_EXPANSION
    }
    # Also include zero-match entities flagged for expansion
    for entity, matches in zip(all_entities, nested):
        if not matches:
            from ofac_matcher.entity_flagger import flag_entity
            flag, _ = flag_entity(entity, [])
            if flag == EntityFlag.NEEDS_CONTEXT_EXPANSION:
                context_entities.add(entity.text)

    if context_entities:
        for text in sorted(context_entities):
            # Find the flag_reason from the flat results
            reasons = [r.flag_reason for r in all_flat if r.comprehend_entity.text == text]
            reason = reasons[0] if reasons else "no_match_single_token"
            print(f"\n  ⚠  {text!r}")
            print(f"       → {reason}")
            print(f"       → Action: re-run Comprehend on expanded surrounding context")
    else:
        print("\n  (none)")


if __name__ == "__main__":
    run_demo()
