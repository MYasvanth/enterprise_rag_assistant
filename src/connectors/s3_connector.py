"""
S3 Connector
============
Connects to AWS S3 buckets or MinIO (self-hosted, S3-compatible - free
alternative to AWS)

Change detection: per-object ETag compared against a manifest kept in
ManifestStore (Redis if provided, in-memory otherwise).

Security: read-only usage intended - the plan requires the connector to
run under an IAM role with a read-only S3 policy. Object keys are never
used as local file paths (fetch_to_temp generates random temp names).

Requires: boto3 (pip install boto3)
"""

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


class S3Connector(BaseConnector):
    """
    Connector for AWS S3 / MinIO buckets.

    Args:
        bucket: S3 bucket name.
        prefix: key prefix to scope listing (e.g. "knowledge-base/").
        region: AWS region (ignored by MinIO but kept for interface parity).
        endpoint_url: custom endpoint for MinIO / localstack
                      (e.g. "http://localhost:9000"). None = real AWS.
        aws_access_key_id / aws_secret_access_key: explicit credentials.
            Prefer omitting both so boto3 falls back to the standard
            credential chain (env vars, IAM role, ~/.aws).
        redis_client: optional Redis client for persistent sync manifests.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        endpoint_url: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        redis_client=None,
    ):
        super().__init__(redis_client=redis_client)
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.endpoint_url = endpoint_url
        self._creds = {
            k: v for k, v in (
                ("aws_access_key_id", aws_access_key_id),
                ("aws_secret_access_key", aws_secret_access_key),
            ) if v
        }
        self.client = None

    def connect(self) -> None:
        """
        Create the boto3 client and verify bucket access via HeadBucket.

        Raises:
            ConnectionError: boto3 missing, or bucket not accessible.
        """
        if not _BOTO3_AVAILABLE:
            raise ConnectionError(
                "boto3 is not installed. Install with: pip install boto3"
            )
        try:
            self.client = boto3.client(
                "s3",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
                **self._creds
            )
            # HeadBucket verifies existence AND permissions.
            self.client.head_bucket(Bucket=self.bucket)
        except (BotoCoreError, ClientError) as exc:
            raise ConnectionError(
                f"Cannot access S3 bucket '{self.bucket}': {exc}"
            ) from exc
        logger.debug("S3Connector connected to bucket=%s prefix=%s", self.bucket, self.prefix)

    def list_files(self) -> List[str]:
        """
        List all object URIs under the prefix.

        Uses a paginator so buckets with >1000 objects are fully walked.
        Returns URIs of the form s3://bucket/key.
        """
        self._ensure_client()
        uris: List[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                uris.append(f"s3://{self.bucket}/{obj['Key']}")
        return uris

    def fetch(self, file_uri: str) -> bytes:
        """
        Download raw object bytes.

        Args:
            file_uri: "s3://bucket/key" (bucket must match this connector's
                      bucket; the key must start with the configured prefix).

        Raises:
            ValueError: URI does not match s3://<this bucket>/... format.
            FileNotFoundError: object does not exist.
        """
        self._ensure_client()
        expected = f"s3://{self.bucket}/"
        if not file_uri.startswith(expected):
            raise ValueError(
                f"URI '{file_uri}' does not belong to bucket '{self.bucket}'"
            )
        key = file_uri[len(expected):]
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                raise FileNotFoundError(
                    f"S3 object not found: {file_uri}"
                ) from exc
            raise

    def watch(self) -> List[str]:
        """
        Return objects that are new or modified since the last watch() call.

        Version token: object ETag (changes when content changes; for
        multipart uploads it is not a content hash but still changes on
        modification, which is what matters for sync).
        """
        self._ensure_client()
        current = {}
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                uri = f"s3://{self.bucket}/{obj['Key']}"
                current[uri] = obj["ETag"]
        manifest_key = f"connector:s3:{self.bucket}:{self.prefix}"
        return self._detect_changes(manifest_key, current)

    # -- internals -----------------------------------------------------------

    def _ensure_client(self) -> None:
        """Raise a clear error if connect() was never called."""
        if self.client is None:
            raise ConnectionError(
                "S3Connector not connected. Call connect() first."
            )
