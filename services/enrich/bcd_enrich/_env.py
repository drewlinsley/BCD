"""The API's `.env` reader, for the enrich commands run from the repo root."""

from __future__ import annotations

import os


def load_dotenv(path: str = ".env") -> None:
    """Put `.env` into the environment, without overriding what is already set.

    `python -m bcd_enrich ...` from the repo root then finds the same catalog the API serves.
    Without it the store falls back to the SQLite dev files under `./data`, and a run there
    would enrich nothing anyone scans.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("\"'")
