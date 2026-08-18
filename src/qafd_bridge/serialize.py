"""Serialize QAFD retrieval traces for consumption by GraphKV.

This module intentionally has no QAFD or GraphKV imports.  It is the boundary
between the two environments: QAFD produces JSONL, and GraphKV reads it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Passage:
    pid: str
    text: str
    qafd_score: float


@dataclass(frozen=True)
class PassageEdge:
    src: int
    dst: int
    entity_hops: int
    path: tuple[str, ...]


@dataclass(frozen=True)
class QAFDRecord:
    qid: str
    question: str
    answer: Any
    passages: tuple[Passage, ...]
    edges: tuple[PassageEdge, ...]
    diameter: int | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passages"] = [asdict(item) for item in self.passages]
        value["edges"] = [
            {**asdict(item), "path": list(item.path)} for item in self.edges
        ]
        return value


def edge_from_path(src: int, dst: int, path: Sequence[str]) -> PassageEdge:
    """Build an edge from the entity path returned by bounded BFS."""
    normalized = tuple(str(node) for node in path)
    if len(normalized) < 2:
        raise ValueError("an entity path must contain at least two entities")
    return PassageEdge(
        src=src,
        dst=dst,
        entity_hops=len(normalized) - 1,
        path=normalized,
    )


def _validate(record: QAFDRecord) -> None:
    if not record.qid:
        raise ValueError("qid must not be empty")
    if not record.question:
        raise ValueError("question must not be empty")
    passage_count = len(record.passages)
    for passage in record.passages:
        if not passage.pid or not passage.text:
            raise ValueError("each passage needs a pid and text")
        if not math.isfinite(float(passage.qafd_score)):
            raise ValueError("qafd_score must be finite")
    for edge in record.edges:
        if not 0 <= edge.src < passage_count or not 0 <= edge.dst < passage_count:
            raise ValueError("edge endpoint is outside the passage list")
        if edge.src == edge.dst:
            raise ValueError("self-edges are not valid passage edges")
        if edge.entity_hops < 1 or len(edge.path) != edge.entity_hops + 1:
            raise ValueError("entity_hops must equal len(path) - 1")
    if record.diameter is not None and record.diameter < 0:
        raise ValueError("diameter must be non-negative or None")


def write_jsonl(records: Iterable[QAFDRecord], output_path: str | Path) -> int:
    """Validate and write one QAFD record per JSONL line.

    Returns the number of records written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            _validate(record)
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(input_path: str | Path) -> list[dict[str, Any]]:
    """Read serialized records without importing either upstream project."""
    with Path(input_path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
