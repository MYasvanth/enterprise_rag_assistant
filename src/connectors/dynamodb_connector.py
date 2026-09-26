"""
DynamoDB Connector
==================
Reads items from a DynamoDB table and serialises each item to a JSON text
document for ingestion - structured data source support.

"Files" are table items, addressed as dynamodb://<table>/<primary_key>.

Change detection: per-item content hash compared against a manifest kept
in ManifestStore (Redis if provided, in-memory otherwise). Note: watch()
scans the table, so it is intended for moderate-size tables; for very
large tables prefer a DynamoDB Streams based sync (documented follow-up).

Requires: boto3 (pip install boto3)
"""

import hashlib
import json
import logging
from typing import List

from src.connectors.base_connector import BaseConnector

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3_AVAILABLE = True
except ImportError:  # pragma: no cover - boto3 optional in dev
    boto3 = None  # type: ignore[assignment]
    BotoCoreError = ClientError = Exception  # type: ignore[assignment,misc]
    _BOTO3_AVAILABLE = False


class DynamoDBConnector(BaseConnector):
    """
    Connector for a DynamoDB table.

    Args:
        table_name: DynamoDB table name.
        region: AWS region.
        endpoint_url: custom endpoint for localstack
                      (e.g. "http://localhost:4566"). None = real AWS.
        primary_key: attribute name of the table's partition key.
        redis_client: optional Redis client for persistent sync manifests.
    """

    def __init__(
        self,
        table_name: str,
        region: str = "us-east-1",
        endpoint_url: str = None,
        primary_key: str = "id",
        redis_client=None,
    ):
        super().__init__(redis_client=redis_client)
        self.table_name = table_name
        self.region = region
        self.endpoint_url = endpoint_url
        self.primary_key = primary_key
        self.table = None

    def connect(self) -> None:
        """
        Create the boto3 table resource and verify the table exists.

        Raises:
            ConnectionError: boto3 missing, or table not accessible.
        """
        if not _BOTO3_AVAILABLE:
            raise ConnectionError(
                "boto3 is not installed. Install with: pip install boto3"
            )
        try:
            resource = boto3.resource(
                "dynamodb",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
            )
            self.table = resource.Table(self.table_name)
            # Force a round trip to verify the table exists and is accessible.
            self.table.load()
        except (BotoCoreError, ClientError) as exc:
            raise ConnectionError(
                f"Cannot access DynamoDB table '{self.table_name}': {exc}"
            ) from exc
        logger.debug("DynamoDBConnector connected to table=%s", self.table_name)

    def list_files(self) -> List[str]:
        """
        List all item URIs in the table.

        Returns URIs of the form dynamodb://<table>/<primary_key_value>.
        """
        self._ensure_connected()
        return [
            self._item_uri(item[self.primary_key])
            for item in self._scan_items()
        ]

    def fetch(self, file_uri: str) -> bytes:
        """
        Fetch a single item and return it as pretty-printed JSON bytes.

        Args:
            file_uri: "dynamodb://<table>/<primary_key_value>".

        Raises:
            ValueError: URI does not match this connector's table.
            FileNotFoundError: item does not exist.
        """
        self._ensure_connected()
        pk_value = self._parse_pk(file_uri)
        response = self.table.get_item(Key={self.primary_key: pk_value})
        item = response.get("Item")
        if item is None:
            raise FileNotFoundError(f"DynamoDB item not found: {file_uri}")
        return json.dumps(item, default=str, indent=2).encode("utf-8")

    def watch(self) -> List[str]:
        """
        Return items that are new or modified since the last watch() call.

        Version token: SHA256 of the item's canonical JSON. This is exact
        but requires a full scan - fine for moderate tables.
        """
        self._ensure_connected()
        current = {}
        for item in self._scan_items():
            uri = self._item_uri(item[self.primary_key])
            canonical = json.dumps(item, default=str, sort_keys=True)
            current[uri] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        manifest_key = f"connector:dynamodb:{self.table_name}"
        return self._detect_changes(manifest_key, current)

    # -- internals -----------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Raise a clear error if connect() was never called."""
        if self.table is None:
            raise ConnectionError(
                "DynamoDBConnector not connected. Call connect() first."
            )

    def _scan_items(self) -> List[dict]:
        """Full table scan with pagination (handles >1MB result sets)."""
        items: List[dict] = []
        response = self.table.scan()
        items.extend(response.get("Items", []))
        while "LastEvaluatedKey" in response:
            response = self.table.scan(
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            items.extend(response.get("Items", []))
        return items

    def _item_uri(self, pk_value) -> str:
        return f"dynamodb://{self.table_name}/{pk_value}"

    def _parse_pk(self, file_uri: str):
        expected = f"dynamodb://{self.table_name}/"
        if not file_uri.startswith(expected):
            raise ValueError(
                f"URI '{file_uri}' does not belong to table '{self.table_name}'"
            )
        raw = file_uri[len(expected):]
        # DynamoDB keys may be numeric; try int first, fall back to string.
        try:
            return int(raw)
        except ValueError:
            return raw

    def _validate_uri(self, file_uri: str) -> None:
        """Sanity-check a DynamoDB item URI before use."""
        self._parse_pk(file_uri)

    def __repr__(self) -> str:
        return (
            f"DynamoDBConnector(table={self.table_name!r}, "
            f"region={self.region!r}, primary_key={self.primary_key!r})"
        )
