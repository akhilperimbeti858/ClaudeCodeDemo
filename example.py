"""
example.py — demonstrates the OFAC fuzzy + semantic matching pipeline.

Run:
    python example.py

The example uses an in-memory SDN stub so no file download is needed.
Swap `OFACMatcher.from_sdn_records(...)` for `OFACMatcher.from_sdn_file("sdn.xml")`
to use the real OFAC list.
"""

import logging

from ofac_matcher import (
    DistanceMetric,
    OFACMatcher,
    PipelineConfig,
    from_raw_list,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# 1. Build a small stub SDN list (replace with load_sdn_list("sdn.xml") in prod)
# ---------------------------------------------------------------------------

SDN_STUB = [
    {
        "uid": "1001",
        "name": "IRAN IMPORT AND EXPORT BANK",
        "entity_type": "Entity",
        "aliases": ["IIMB", "Iran Import Bank"],
        "programs": ["IRAN"],
    },
    {
        "uid": "1002",
        "name": "AL-QAIDA",
        "entity_type": "Entity",
        "aliases": ["AL QAEDA", "AL QA'IDA", "AL-QAEDA"],
        "programs": ["SDGT"],
    },
    {
        "uid": "1003",
        "name": "KIM JONG UN",
        "entity_type": "Individual",
        "aliases": ["KIM JONGUN", "KIM, JONG UN"],
        "programs": ["DPRK"],
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
]

# ---------------------------------------------------------------------------
# 2. Simulate AWS Comprehend detect_entities() output
#    (as returned by boto3 comprehend.detect_entities())
# ---------------------------------------------------------------------------

COMPREHEND_RESPONSE = {
    "Entities": [
        {
            "Text": "Iran Import Bank",
            "Type": "ORGANIZATION",
            "Score": 0.9987,
            "BeginOffset": 12,
            "EndOffset": 28,
        },
        {
            "Text": "Al-Qaeda",         # variant spelling
            "Type": "ORGANIZATION",
            "Score": 0.9721,
            "BeginOffset": 45,
            "EndOffset": 53,
        },
        {
            "Text": "Kim Jong-un",       # hyphenated, will fuzz-match well
            "Type": "PERSON",
            "Score": 0.9944,
            "BeginOffset": 65,
            "EndOffset": 76,
        },
        {
            "Text": "Mahan Airways",     # alias variant
            "Type": "ORGANIZATION",
            "Score": 0.8831,
            "BeginOffset": 90,
            "EndOffset": 103,
        },
        {
            "Text": "ACME Corporation",  # deliberate non-match
            "Type": "ORGANIZATION",
            "Score": 0.7200,
            "BeginOffset": 120,
            "EndOffset": 136,
        },
    ],
    "ResponseMetadata": {"HTTPStatusCode": 200},
}


def run_demo() -> None:
    # ------------------------------------------------------------------
    # 3. Configure the pipeline
    # ------------------------------------------------------------------
    config = PipelineConfig(
        fuzzy_threshold=65.0,
        fuzzy_top_k=25,
        semantic_threshold=0.40,
        semantic_top_k=5,
        distance_metric=DistanceMetric.L2,   # swap to DistanceMetric.L1 for Manhattan
        embedding_model="all-MiniLM-L6-v2",
        fuzzy_weight=0.35,
        semantic_weight=0.65,
        # Only match PERSON / ORGANIZATION entities from Comprehend:
        entity_type_filter=["PERSON", "ORGANIZATION"],
        # use_tfidf_fallback=False  →  uses sentence-transformers (requires internet/local model)
        # use_tfidf_fallback=True   →  offline TF-IDF char n-gram fallback
        use_tfidf_fallback=True,
    )

    # ------------------------------------------------------------------
    # 4. Build the matcher (indexes the SDN corpus once)
    # ------------------------------------------------------------------
    matcher = OFACMatcher.from_sdn_records(SDN_STUB, config=config)

    # ------------------------------------------------------------------
    # 5a. Match — nested output (one list per query entity)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("NESTED RESULTS  (one block per Comprehend entity)")
    print("=" * 70)

    nested = matcher.match(COMPREHEND_RESPONSE, input_format="detect")

    for entity_matches in nested:
        if not entity_matches:
            continue
        query = entity_matches[0].comprehend_entity
        print(f"\nQuery: {query.text!r}  [{query.entity_type}  conf={query.score:.2%}]")
        print(f"  {'Rank':<5} {'SDN Name':<40} {'Combined':>8} {'Fuzzy':>7} {'Cosine':>8} {'Dist':>8}  Alias")
        print("  " + "-" * 90)
        for r in entity_matches:
            print(
                f"  {r.rank:<5} {r.sdn_entity.name:<40} "
                f"{r.combined_score:>8.4f} "
                f"{r.layer_score.fuzzy_score:>7.1f} "
                f"{r.layer_score.semantic_score:>8.4f} "
                f"{r.layer_score.distance:>8.4f}  "
                f"{r.layer_score.matched_alias!r}"
            )

    # ------------------------------------------------------------------
    # 5b. Match — flat output, minimum combined score 0.6
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FLAT RESULTS  (all entities, combined_score >= 0.60)")
    print("=" * 70)

    flat = matcher.match_flat(COMPREHEND_RESPONSE, min_combined_score=0.60)
    print(f"\n{'Query':<25} {'SDN Name':<40} {'Combined':>8}")
    print("-" * 78)
    for r in flat:
        print(
            f"{r.comprehend_entity.text:<25} "
            f"{r.sdn_entity.name:<40} "
            f"{r.combined_score:>8.4f}"
        )

    # ------------------------------------------------------------------
    # 5c. Pass raw entity dicts directly (no boto3 response wrapper needed)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RAW LIST INPUT")
    print("=" * 70)

    raw_entities = [
        {"Text": "Sberbank", "Type": "ORGANIZATION", "Score": 0.91},
        {"Text": "Hizballah", "Type": "ORGANIZATION", "Score": 0.88},
        {"Text": "Wagner PMC", "Type": "ORGANIZATION", "Score": 0.79},
    ]
    flat2 = matcher.match_flat(raw_entities, input_format="raw_list", min_combined_score=0.50)
    for r in flat2:
        print(
            f"  {r.comprehend_entity.text!r:25s} → {r.sdn_entity.name!r}  "
            f"(combined={r.combined_score:.4f})"
        )


if __name__ == "__main__":
    run_demo()
