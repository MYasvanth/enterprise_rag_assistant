"""
Local Filesystem Connector
==========================
Reads supported files (PDF, DOCX, TXT, MD) from a local directory tree.
This is the current behaviour of the system - manual/local ingestion -
used in dev and for manual uploads via the API

Change detection: per-file mtime compared against a manifest kept in
ManifestStore (Redis if provided, in-memory otherwise).

Security: path traversal protection - every fetch resolves the path and
verifies it stays inside base_path.
"""

import logging
from pathlib import Path
from typing import List, Optional

from src.connectors.base_connector import SUPPORTED_EXTENSIONS, BaseConnector

logger = logging.getLogger(__name__)


class LocalConnector(BaseConnector):
    """Connector for the local filesystem."""

    def __init__(self, base_path: str, redis_client=None):
        super().__init__(redis_client=redis_client)
        self.base_path = Path(base_path).resolve()

    def connect(self) -> None:
        """Verify the base path exists and is a directory."""
        if not self.base_path.exists():
            raise ConnectionError(f"Path not found: {self.base_path}")
        if not self.base_path.is_dir():
            raise ConnectionError(f"Path is not a directory: {self.base_path}")
        logger.debug("LocalConnector connected to %s", self.base_path)

    def list_files(self) -> List[str]:
        """List all supported files under base_path (recursive)."""
        return [
            str(p.resolve())
            for p in self.base_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    def fetch(self, file_uri: str) -> bytes:
        """
        Read raw file bytes.

        Raises:
            ValueError: path traversal attempt (resolved path escapes base_path).
            FileNotFoundError: file does not exist.
        """
        path = Path(file_uri).resolve()
        # Traversal protection - resolved path must stay within base_path.
        # is_relative_to is stronger than startswith (rejects sibling dirs
        # that merely share a path prefix).
        if not path.is_relative_to(self.base_path):
            raise ValueError(f"Path traversal attempt: {file_uri}")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_bytes()

    def watch(self) -> List[str]:
        """
        Return files that are new or modified since the last watch() call.

        Version token: file mtime (as string). State kept in ManifestStore.
        """
        current = {
            f: str(Path(f).stat().st_mtime)
            for f in self.list_files()
        }
        manifest_key = f"connector:local:{self.base_path}"
        return self._detect_changes(manifest_key, current)
