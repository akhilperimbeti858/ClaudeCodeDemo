"""
Loader for the OFAC Specially Designated Nationals (SDN) list.

Supports:
  - Official OFAC XML format  (sdn.xml  / consolidated.xml)
  - OFAC CSV flat format      (sdn.csv)

Download sources (public):
  https://ofac.treasury.gov/system/files/sdn.xml
  https://ofac.treasury.gov/system/files/sdn.csv
"""

from __future__ import annotations

import csv
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

from .models import EntityType, SDNEntity

logger = logging.getLogger(__name__)

# OFAC XML namespace
_NS = {"sdn": "https://sanctionslistservice.ofac.treas.gov/api/PublicationsService/data/1"}


def _norm_entity_type(raw: str) -> EntityType:
    mapping = {
        "individual": EntityType.INDIVIDUAL,
        "entity": EntityType.ENTITY,
        "vessel": EntityType.VESSEL,
        "aircraft": EntityType.AIRCRAFT,
    }
    return mapping.get(raw.lower().strip(), EntityType.UNKNOWN)


# ---------------------------------------------------------------------------
# XML loader
# ---------------------------------------------------------------------------

def _parse_xml(path: Path) -> list[SDNEntity]:
    """Parse the official OFAC SDN XML file."""
    tree = ET.parse(path)
    root = tree.getroot()

    # Handle both namespaced and bare XML
    ns_prefix = ""
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0].lstrip("{")
        ns_prefix = f"{{{ns_uri}}}"

    entities: list[SDNEntity] = []

    sdn_entries = root.findall(f".//{ns_prefix}sdnEntry")
    if not sdn_entries:
        # Try without namespace (older format)
        sdn_entries = root.findall(".//sdnEntry")

    for entry in sdn_entries:
        def _text(tag: str) -> str:
            el = entry.find(f"{ns_prefix}{tag}") or entry.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        uid = _text("uid")
        last_name = _text("lastName")
        first_name = _text("firstName")
        sdk_type = _text("sdnType")

        if first_name:
            name = f"{first_name} {last_name}".strip()
        else:
            name = last_name

        if not name:
            continue

        # Collect AKA aliases
        aliases: list[str] = []
        aka_list = entry.find(f"{ns_prefix}akaList") or entry.find("akaList")
        if aka_list is not None:
            for aka in aka_list:
                aka_last = aka.find(f"{ns_prefix}lastName") or aka.find("lastName")
                aka_first = aka.find(f"{ns_prefix}firstName") or aka.find("firstName")
                last_t = aka_last.text.strip() if aka_last is not None and aka_last.text else ""
                first_t = aka_first.text.strip() if aka_first is not None and aka_first.text else ""
                alias = f"{first_t} {last_t}".strip() if first_t else last_t
                if alias and alias.lower() != name.lower():
                    aliases.append(alias)

        # Programs
        programs: list[str] = []
        prog_list = entry.find(f"{ns_prefix}programList") or entry.find("programList")
        if prog_list is not None:
            for prog in prog_list:
                if prog.text:
                    programs.append(prog.text.strip())

        entities.append(
            SDNEntity(
                uid=uid,
                name=name,
                entity_type=_norm_entity_type(sdk_type),
                aliases=aliases,
                programs=programs,
            )
        )

    logger.info("Loaded %d SDN entries from XML: %s", len(entities), path)
    return entities


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _parse_csv(path: Path) -> list[SDNEntity]:
    """
    Parse the OFAC flat CSV.

    Columns (0-indexed):
      0  ent_num, 1  SDN_Name, 2  SDN_Type, 3  Program, 4  Title,
      5  Call_Sign, 6  Vess_type, 7  Tonnage, 8  GRT, 9  Vess_flag,
      10 Vess_owner, 11 Remarks
    """
    entities: list[SDNEntity] = []
    grouped: dict[str, SDNEntity] = {}

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            uid = row[0].strip()
            raw_name = row[1].strip()
            sdk_type = row[2].strip()

            if not raw_name or raw_name == "-0-":
                continue

            name = raw_name.rstrip(";").strip()

            if uid not in grouped:
                grouped[uid] = SDNEntity(
                    uid=uid,
                    name=name,
                    entity_type=_norm_entity_type(sdk_type),
                    programs=[row[3].strip()] if len(row) > 3 and row[3].strip() else [],
                )
            else:
                # Repeated UID rows indicate additional programs or aliases
                entity = grouped[uid]
                if len(row) > 3 and row[3].strip():
                    entity.programs.append(row[3].strip())

    entities = list(grouped.values())

    # Re-build all_names after construction
    for e in entities:
        e.all_names = [e.name] + e.aliases

    logger.info("Loaded %d SDN entries from CSV: %s", len(entities), path)
    return entities


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_sdn_list(path: Union[str, Path]) -> list[SDNEntity]:
    """
    Load the OFAC SDN list from *path*.

    Auto-detects XML vs CSV by file extension.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SDN file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".xml":
        return _parse_xml(path)
    elif suffix == ".csv":
        return _parse_csv(path)
    else:
        raise ValueError(f"Unsupported SDN file format: {suffix!r} (expected .xml or .csv)")


def load_sdn_from_dict(records: list[dict]) -> list[SDNEntity]:
    """
    Build SDN entities from a plain list of dicts — useful for testing or
    when the list is already ingested from a database.

    Expected keys: uid, name, entity_type (str), aliases (list[str]), programs (list[str])
    """
    entities = []
    for r in records:
        entities.append(
            SDNEntity(
                uid=str(r.get("uid", "")),
                name=r["name"],
                entity_type=_norm_entity_type(r.get("entity_type", "")),
                aliases=r.get("aliases", []),
                programs=r.get("programs", []),
            )
        )
    return entities
