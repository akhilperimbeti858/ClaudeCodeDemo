"""
Parser for AWS Comprehend custom entity recognition results.

Handles two response shapes:
  1. Real-time / synchronous  — response from detect_entities()
  2. Batch / async job output — line-delimited JSON from S3 output

Both shapes ultimately produce a list of ComprehendEntity objects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

from .models import ComprehendEntity

logger = logging.getLogger(__name__)


def _entity_from_dict(raw: dict) -> ComprehendEntity:
    return ComprehendEntity(
        text=raw.get("Text", raw.get("text", "")).strip(),
        entity_type=raw.get("Type", raw.get("type", "UNKNOWN")).upper(),
        score=float(raw.get("Score", raw.get("score", 0.0))),
        begin_offset=int(raw.get("BeginOffset", raw.get("beginOffset", 0))),
        end_offset=int(raw.get("EndOffset", raw.get("endOffset", 0))),
    )


def parse_detect_entities_response(response: dict) -> list[ComprehendEntity]:
    """
    Parse the dict returned by boto3's ``comprehend.detect_entities()``.

    Example input::

        {
            "Entities": [
                {"Text": "IRAN IMPORT BANK", "Type": "ORGANIZATION", "Score": 0.9987,
                 "BeginOffset": 0, "EndOffset": 16}
            ],
            "ResponseMetadata": {...}
        }
    """
    raw_entities = response.get("Entities", [])
    entities = [_entity_from_dict(e) for e in raw_entities if e.get("Text") or e.get("text")]
    logger.debug("Parsed %d entities from detect_entities response", len(entities))
    return entities


def parse_batch_response(response: dict) -> list[ComprehendEntity]:
    """
    Parse the dict returned by boto3's ``comprehend.batch_detect_entities()``.

    Each item in ``ResultList`` has an ``Entities`` sub-list.
    """
    all_entities: list[ComprehendEntity] = []
    for item in response.get("ResultList", []):
        all_entities.extend(parse_detect_entities_response(item))
    logger.debug("Parsed %d entities from batch_detect_entities response", len(all_entities))
    return all_entities


def parse_async_job_output(path: Union[str, Path]) -> list[ComprehendEntity]:
    """
    Parse the line-delimited JSON output file produced by a Comprehend async
    analysis job (downloaded from S3).

    Each line is either:
      - A ``detect_entities`` response dict (has ``"Entities"`` key), or
      - A raw entity dict (has ``"Text"`` / ``"Score"`` keys directly).
    """
    path = Path(path)
    all_entities: list[ComprehendEntity] = []

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_no, exc)
                continue

            if "Entities" in obj or "entities" in obj:
                all_entities.extend(parse_detect_entities_response(obj))
            elif "Text" in obj or "text" in obj:
                all_entities.append(_entity_from_dict(obj))
            else:
                logger.debug("Line %d: unrecognised shape, skipping", line_no)

    logger.info("Parsed %d entities from async job file: %s", len(all_entities), path)
    return all_entities


def from_raw_list(records: list[dict]) -> list[ComprehendEntity]:
    """
    Build ComprehendEntity objects from a plain list of dicts.
    Accepts both camelCase (AWS SDK) and snake_case keys.

    Useful for passing test fixtures or pre-processed data directly
    into the pipeline without going through boto3.
    """
    return [_entity_from_dict(r) for r in records if r.get("Text") or r.get("text")]
