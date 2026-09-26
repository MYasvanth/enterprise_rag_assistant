"""
Base Connector Interface
========================
Abstract base class that every data source connector implements.
Part of Layer 1 - Data Sources (docs/Bits/01_layer1_data_sources.md,
docs/Bits/LLD_01_layer1.md).

Every connector returns raw file bytes via fetch() and detects new or
modified files via watch(), regardless of where the data lives
(local disk, S3/MinIO, SharePoint, Confluence, DynamoDB). The ingestion
pipeline never needs to know the source.

Lifecycle:
    connector = S3Connector(bucket="docs", prefix="kb/", region="us-east-1")
    connector.connect()                  # raise ConnectionError if unreachable
    for uri in connector.watch():        # new/modified since last sync
        raw_bytes = connector.fetch(uri) # download bytes
    everything = connector.list_files()  # full listing

Sync state for watch() is kept in a ManifestStore. When a Redis client is
provided it is used (persistent across restarts); otherwise an in-process
dict is used so connectors work out of the box in dev.
"""

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# File extensions the ingestion pipeline can parse (Layer 1 target formats).
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class ManifestStore:
    """
    Stores per-connector sync manifests (file URI -> version token) used by
    watch() to detect new or modified files.

    Uses Redis when a client is provided (persistent across restarts) and
    falls back to an in-process dict otherwise. The fallback means
    connectors work out of the box in dev; Redis is recommended for
    production so sync state survives worker restarts.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._memory: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        if self._redis is not None:
            raw = self._redis.get(key)
            if isinstance(raw, bytes):
                return raw.decode("utf-8")
            return raw
        return self._memory.get(key)

    def set(self, key: str, value: str) -> None:
        if self._redis is not None:
            self._redis.set(key, value)
        else:
            self._memory[key] = value


class BaseConnector(ABC):
    """Abstract base class for all data source connectors."""

    def __init__(self, redis_client=None):
        # Each connector subclass passes through an optional Redis client.
        # Without Redis, sync manifests are kept in memory (dev mode).
        self.manifest = ManifestStore(redis_client)

    # -- Required interface ------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source. Raise ConnectionError on failure."""

    @abstractmethod
    def list_files(self) -> List[str]:
        """Return list of file paths/URIs available in source."""

    @abstractmethod
    def fetch(self, file_uri: str) -> bytes:
        """Download raw file bytes. Raise FileNotFoundError if missing."""

    @abstractmethod
    def watch(self) -> List[str]:
        """
        Return list of new/modified file URIs since last sync.

        Implementations compare a per-file version token (mtime, ETag,
        page version, content hash) against the manifest store, then
        update the manifest. On the very first call everything is
        reported as new (initial sync). Deleted-file detection is not
        part of this contract (documented follow-up in the plan).
        """

    # -- Shared helpers ------------------------------------------------------

    def _detect_changes(self, manifest_key: str, current: Dict[str, str]) -> List[str]:
        """
        Diff current version tokens against the stored manifest.

        Args:
            manifest_key: unique key for this connector's manifest
                          (e.g. "connector:s3:bucket:prefix").
            current: mapping of file URI -> version token (mtime/ETag/hash).

        Returns:
            URIs that are new or whose version token changed. The manifest
            is updated as part of the call.
        """
        stored_raw = self.manifest.get(manifest_key)
        stored: Dict[str, str] = json.loads(stored_raw) if stored_raw else {}

        new_or_modified = [
            uri for uri, token in current.items()
            if uri not in stored or stored[uri] != token
        ]

        self.manifest.set(manifest_key, json.dumps(current))

        if new_or_modified:
            logger.info(
                "%s detected %d new/modified file(s)",
                self.__class__.__name__,
                len(new_or_modified),
            )
        return new_or_modified

    def fetch_to_temp(self, file_uri: str) -> str:
        """
        Fetch a file and save it to a temp path with a RANDOM name
        (never the original filename - path traversal fix from the plan).

        The original file extension is preserved as the temp suffix so the
        ingestion pipeline can select the right parser. Caller is
        responsible for deleting the temp file after processing.

        Returns:
            Path to the temp file.
        """
        content = self.fetch(file_uri)
        suffix = os.path.splitext(file_uri)[1] or ".tmp"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(content)
        finally:
            tmp.close()
        logger.debug("Fetched %s -> %s (%d bytes)", file_uri, tmp.name, len(content))
        return tmp.name
