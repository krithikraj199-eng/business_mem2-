"""Build the SQLite byte-offset index for normalized Member 1 TSV files.

Committed implementation from https://github.com/Harish-121-ss/aws_mem1
Branch: member1-disk-lookup
Commit: 97bede1655a5400fcda623344fd5d05b4725287f
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
from pathlib import Path

try:
    from .disk_lookup import SOURCE_COLUMNS, SOURCE_FILES, SCHEMA_VERSION
except ImportError:
    from disk_lookup import SOURCE_COLUMNS, SOURCE_FILES, SCHEMA_VERSION

DEFAULT_DATA_DIR = Path("data") / "normalized"
DEFAULT_INDEX = Path("data") / "lookup" / "entity_lookup.sqlite3"


def _read_header(fh, path: Path) -> bytes:
    raw = fh.readline()
    if not raw:
        raise ValueError(f"Empty normalized TSV: {path}")
    try:
        text = raw.decode("utf-8-sig").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 header: {path}") from exc
    header = next(csv.reader([text], delimiter="\t"))
    if header != SOURCE_COLUMNS:
        raise ValueError(
            f"Unexpected schema in {path}. "
            f"Expected {SOURCE_COLUMNS}, found {header}"
        )
    return raw


def _validate_record(raw: bytes, path: Path, line_number: int) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Invalid UTF-8 in {path} line {line_number}"
        ) from exc
    values = next(csv.reader([text], delimiter="\t"))
    if len(values) != len(SOURCE_COLUMNS):
        raise ValueError(
            f"Invalid TSV field count in {path} line {line_number}: "
            f"expected {len(SOURCE_COLUMNS)}, found {len(values)}"
        )
    entity_id = values[0]
    if not entity_id:
        raise ValueError(f"Empty entity_id in {path} line {line_number}")
    return entity_id


def build_lookup_index(data_dir: Path, index_path: Path) -> None:
    """Build a fresh SQLite index and atomically replace the target file."""
    data_dir = data_dir.resolve()
    index_path = index_path.resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)

    conn = sqlite3.connect(temp_path)
    try:
        conn.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE source_files (
                source TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL
            );
            CREATE TABLE entity_index (
                entity_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                byte_offset INTEGER NOT NULL
            );
            CREATE INDEX idx_entity_source
                ON entity_index(source);
            """
        )
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", str(SCHEMA_VERSION)),
                ("data_dir", str(data_dir)),
                ("source_columns", "\t".join(SOURCE_COLUMNS)),
                ("encoding", "utf-8"),
                ("newline_mode", "binary readline; CRLF/LF supported"),
            ],
        )

        total = 0
        for source, filename in SOURCE_FILES.items():
            path = data_dir / filename
            if not path.exists():
                raise FileNotFoundError(path)

            stat_before = path.stat()
            digest = hashlib.sha256()
            inserted = 0

            with path.open("rb") as fh:
                header = _read_header(fh, path)
                digest.update(header)
                while True:
                    offset = fh.tell()
                    raw = fh.readline()
                    if not raw:
                        break
                    digest.update(raw)
                    entity_id = _validate_record(
                        raw, path, inserted + 2
                    )
                    try:
                        conn.execute(
                            "INSERT INTO entity_index"
                            "(entity_id, source, byte_offset) "
                            "VALUES (?, ?, ?)",
                            (entity_id, source, offset),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"Duplicate entity_id {entity_id!r} "
                            f"encountered in {path}"
                        ) from exc
                    inserted += 1
                    total += 1

            stat_after = path.stat()
            sha256 = digest.hexdigest()
            if (
                stat_before.st_size != stat_after.st_size
                or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            ):
                raise RuntimeError(
                    f"Source file changed while indexing: {path}"
                )

            conn.execute(
                "INSERT INTO source_files"
                "(source, filename, file_size, mtime_ns, sha256) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    source,
                    filename,
                    stat_after.st_size,
                    stat_after.st_mtime_ns,
                    sha256,
                ),
            )
            print(f"{source}: {inserted:,} records indexed")

        conn.commit()
        conn.close()
        temp_path.replace(index_path)
        print(f"Index: {index_path}")
        print(f"Total records: {total:,}")
    except Exception:
        conn.rollback()
        conn.close()
        temp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    build_lookup_index(args.data_dir, args.index)


if __name__ == "__main__":
    main()
