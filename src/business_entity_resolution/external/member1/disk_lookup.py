"""Read-only disk-backed lookup for normalized Member 1 TSV files.

Committed implementation from https://github.com/Harish-121-ss/aws_mem1
Branch: member1-disk-lookup
Commit: 97bede1655a5400fcda623344fd5d05b4725287f
"""
from __future__ import annotations

import csv
import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable

SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "business_name_normalized",
    "business_name_tokens",
    "business_address_normalized",
    "address_tokens",
    "address_numbers",
    "postal_codes",
    "country_normalized",
]
SOURCE_FILES = {
    "source1": "normalized_source1.tsv",
    "source2": "normalized_source2.tsv",
    "source3": "normalized_source3.tsv",
}
SCHEMA_VERSION = 1


class LookupIntegrityError(RuntimeError):
    """Raised when an indexed TSV no longer matches its build fingerprint."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class DiskEntityLookup:
    """Read-only random-access lookup over normalized TSV files."""

    def __init__(
        self,
        index_path: str | Path,
        data_dir: str | Path | None = None,
        *,
        verify_hash: bool = False,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)
        self._conn = sqlite3.connect(
            f"file:{self.index_path}?mode=ro", uri=True
        )
        self._conn.row_factory = sqlite3.Row
        try:
            meta = self._load_metadata()
            recorded_dir = Path(meta["data_dir"])
            self.data_dir = (
                Path(data_dir).resolve() if data_dir else recorded_dir
            )
            if not self.data_dir.is_absolute():
                self.data_dir = (
                    self.index_path.parent / self.data_dir
                ).resolve()
            self._files = self._load_files()
            self._verify_source_files(hash_files=verify_hash)
        except Exception:
            self._conn.close()
            raise

    def _load_metadata(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM metadata"
        ).fetchall()
        meta = {row["key"]: row["value"] for row in rows}
        if meta.get("schema_version") != str(SCHEMA_VERSION):
            raise LookupIntegrityError(
                f"Unsupported lookup schema version: "
                f"{meta.get('schema_version')}"
            )
        if meta.get("source_columns") != "\t".join(SOURCE_COLUMNS):
            raise LookupIntegrityError("Normalized source schema does not match lookup index")
        return meta

    def _load_files(self) -> dict[str, sqlite3.Row]:
        rows = self._conn.execute(
            "SELECT source, filename, file_size, mtime_ns, sha256 "
            "FROM source_files"
        ).fetchall()
        result = {row["source"]: row for row in rows}
        if set(result) != set(SOURCE_FILES):
            raise LookupIntegrityError(
                "Lookup index does not contain all S1/S2/S3 source files"
            )
        return result

    def _source_path(self, source: str) -> Path:
        return self.data_dir / self._files[source]["filename"]

    def _verify_source_files(self, *, hash_files: bool) -> None:
        for source, row in self._files.items():
            path = self._source_path(source)
            if not path.exists():
                raise LookupIntegrityError(
                    f"Indexed source file missing: {path}"
                )
            stat = path.stat()
            if (
                stat.st_size != row["file_size"]
                or stat.st_mtime_ns != row["mtime_ns"]
            ):
                raise LookupIntegrityError(
                    f"Indexed source file changed: {path}. "
                    "Rebuild the lookup index."
                )
            if hash_files:
                actual = sha256_file(path)
                if actual != row["sha256"]:
                    raise LookupIntegrityError(
                        f"SHA-256 mismatch for indexed source file: {path}"
                    )

    def _lookup_rows(
        self, entity_ids: Iterable[str]
    ) -> list[sqlite3.Row | None]:
        ids = list(entity_ids)
        if not ids:
            return []
        # Stay below SQLite's host-parameter limit for larger bounded batches.
        by_id: dict[str, sqlite3.Row] = {}
        for start in range(0, len(ids), 900):
            batch = ids[start:start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                "SELECT entity_id, source, byte_offset "
                f"FROM entity_index WHERE entity_id IN ({placeholders})",
                batch,
            ).fetchall()
            by_id.update({row["entity_id"]: row for row in rows})
        return [by_id.get(entity_id) for entity_id in ids]

    def get(self, entity_id: str) -> dict[str, str] | None:
        """Return one normalized record, or None if the ID is absent."""
        row = self._conn.execute(
            "SELECT entity_id, source, byte_offset "
            "FROM entity_index WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        return self._read_record(row) if row is not None else None

    def get_many(
        self, entity_ids: Iterable[str]
    ) -> list[dict[str, str] | None]:
        """Return records in exactly the same order as entity_ids."""
        rows = self._lookup_rows(entity_ids)
        return [
            self._read_record(row) if row is not None else None
            for row in rows
        ]

    def _read_record(self, row: sqlite3.Row) -> dict[str, str]:
        path = self._source_path(row["source"])
        with path.open("rb") as fh:
            fh.seek(row["byte_offset"])
            raw = fh.readline()
        if not raw:
            raise LookupIntegrityError(
                f"Indexed offset is outside file: "
                f"{path}:{row['byte_offset']}"
            )
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LookupIntegrityError(
                f"Invalid UTF-8 at {path}:{row['byte_offset']}"
            ) from exc
        values = next(csv.reader([line], delimiter="\t"))
        if len(values) != len(SOURCE_COLUMNS):
            raise LookupIntegrityError(
                f"Indexed record has {len(values)} fields, expected "
                f"{len(SOURCE_COLUMNS)}: {path}:{row['byte_offset']}"
            )
        record = dict(zip(SOURCE_COLUMNS, values))
        if record["entity_id"] != row["entity_id"]:
            raise LookupIntegrityError(
                f"Entity ID mismatch at {path}:{row['byte_offset']}: "
                f"index={row['entity_id']!r}, "
                f"file={record['entity_id']!r}"
            )
        return record

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DiskEntityLookup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
