"""
SharePoint / OneDrive Connector
===============================
Connects to SharePoint document libraries via OAuth2 service account

Auth model (per the plan):
    - Azure AD app registration with client credentials flow
    - Least-privilege service account (Sites.Read.All on the target site)
    - No user passwords, no interactive login

Change detection: per-item Modified datetime compared against a manifest
kept in ManifestStore (Redis if provided, in-memory otherwise).

Requires: office365-rest-python-client (pip install office365-rest-python-client)
"""

import logging
from typing import List

from src.connectors.base_connector import BaseConnector

logger = logging.getLogger(__name__)

try:
    from office365.sharepoint.client_context import ClientContext
    _OFFICE365_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    ClientContext = None  # type: ignore[assignment]
    _OFFICE365_AVAILABLE = False


class SharePointConnector(BaseConnector):
    """
    Connector for SharePoint document libraries.

    Args:
        site_url: full site URL, e.g. "https://tenant.sharepoint.com/sites/KB"
        client_id: Azure AD application (client) ID.
        client_secret: Azure AD application secret.
        library_name: document library title (default "Documents").
        redis_client: optional Redis client for persistent sync manifests.
    """

    def __init__(
        self,
        site_url: str,
        client_id: str,
        client_secret: str,
        library_name: str = "Documents",
        redis_client=None,
    ):
        super().__init__(redis_client=redis_client)
        self.site_url = site_url
        self._client_id = client_id
        self._client_secret = client_secret
        self.library_name = library_name
        self.ctx = None

    def connect(self) -> None:
        """
        Create an authenticated ClientContext and verify site access.

        Raises:
            ConnectionError: library missing, or credentials/site access fail.
        """
        if not _OFFICE365_AVAILABLE:
            raise ConnectionError(
                "office365-rest-python-client is not installed. "
                "Install with: pip install office365-rest-python-client"
            )
        try:
            self.ctx = ClientContext(self.site_url).with_client_credentials(
                self._client_id, self._client_secret
            )
            # Force a round trip to verify credentials and site access.
            self.ctx.web.get().execute_query()
        except Exception as exc:
            raise ConnectionError(
                f"SharePoint connection failed for {self.site_url}: {exc}"
            ) from exc
        logger.debug("SharePointConnector connected to %s", self.site_url)

    def list_files(self) -> List[str]:
        """
        List all files in the document library.

        Returns URIs of the form /sites/<site>/<library>/<file>
        (SharePoint-relative file paths, usable directly by fetch()).
        """
        self._ensure_connected()
        library = self.ctx.web.lists.get_by_title(self.library_name)
        items = library.items.filter("FSObjType eq 0").get().execute_query()
        return [
            item.properties["FileRef"]
            for item in items
            if item.properties.get("FileRef")
        ]

    def fetch(self, file_uri: str) -> bytes:
        """
        Download raw file bytes from the document library.

        Args:
            file_uri: SharePoint-relative file path (as returned by list_files).

        Raises:
            FileNotFoundError: file does not exist or is not readable.
        """
        self._ensure_connected()
        from office365.sharepoint.files.file import File

        response = File.open_binary(self.ctx, file_uri)
        if response.status_code != 200:
            raise FileNotFoundError(
                f"SharePoint file not found ({response.status_code}): {file_uri}"
            )
        return response.content

    def watch(self) -> List[str]:
        """
        Return files that are new or modified since the last watch() call.

        Version token: item Modified datetime (ISO string). This is the
        simplest reliable token exposed by the client; the plan's
        change-token approach can be swapped in here without changing
        the BaseConnector contract.
        """
        self._ensure_connected()
        library = self.ctx.web.lists.get_by_title(self.library_name)
        items = library.items.filter("FSObjType eq 0").get().execute_query()
        current = {}
        for item in items:
            uri = item.properties.get("FileRef")
            if not uri:
                continue
            modified = item.properties.get("Modified", "")
            current[uri] = str(modified)
        manifest_key = f"connector:sharepoint:{self.site_url}:{self.library_name}"
        return self._detect_changes(manifest_key, current)

    # -- internals -----------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Raise a clear error if connect() was never called."""
        if self.ctx is None:
            raise ConnectionError(
                "SharePointConnector not connected. Call connect() first."
            )
