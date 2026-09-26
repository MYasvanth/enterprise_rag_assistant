"""
Data Source Connectors (Layer 1)
================================
Connectors for all supported data sources

Every connector implements the BaseConnector interface:
    connect()   - verify access, raise ConnectionError on failure
    list_files() - full listing of available file URIs
    fetch(uri)  - download raw bytes
    watch()     - new/modified URIs since last sync

Connectors with optional third-party dependencies (boto3,
office365-rest-python-client) are imported lazily so that importing
this package never fails when those dependencies are not installed.
"""

from src.connectors.base_connector import (
    SUPPORTED_EXTENSIONS,
    BaseConnector,
    ManifestStore,
)
from src.connectors.local_connector import LocalConnector

__all__ = [
    "BaseConnector",
    "ManifestStore",
    "SUPPORTED_EXTENSIONS",
    "LocalConnector",
    "S3Connector",
    "SharePointConnector",
    "ConfluenceConnector",
    "DynamoDBConnector",
]


def __getattr__(name):
    """
    Lazy imports for connectors with optional third-party dependencies.

    This keeps `import src.connectors` working even when boto3 or
    office365-rest-python-client are not installed; the ImportError
    only surfaces when the specific connector is actually requested.
    """
    if name == "S3Connector":
        from src.connectors.s3_connector import S3Connector
        return S3Connector
    if name == "SharePointConnector":
        from src.connectors.sharepoint_connector import SharePointConnector
        return SharePointConnector
    if name == "ConfluenceConnector":
        from src.connectors.confluence_connector import ConfluenceConnector
        return ConfluenceConnector
    if name == "DynamoDBConnector":
        from src.connectors.dynamodb_connector import DynamoDBConnector
        return DynamoDBConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
