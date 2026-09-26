"""
Confluence Connector
====================
Reads pages from a Confluence Cloud / Server / Data Center space via the
REST API. Extra enterprise source beyond the original plan - follows the same BaseConnector
interface so the ingestion pipeline treats it identically.

Pages are exported as markdown-like text (title + stripped storage-format
body). "Files" are pages, addressed as confluence://<base_url>/pages/<id>.

Change detection: per-page version number compared against a manifest
kept in ManifestStore (Redis if provided, in-memory otherwise).

Uses `requests` (already in requirements.txt) - no extra dependency.

Auth:
    Cloud:   email + API token
            
    Server:  username + password or personal access token
"""

import html as html_lib
import logging
import re
from typing import Dict, List, Optional

import requests

from src.connectors.base_connector import BaseConnector

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")


def _strip_html(xhtml: str) -> str:
    """
    Minimal Confluence storage-format (XHTML) to plain text conversion.

    Good enough for ingestion: keeps text, drops markup, collapses
    whitespace. Not a full HTML parser - tables become space-separated
    cell text.
    """
    text = _TAG_RE.sub(" ", xhtml or "")
    text = html_lib.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


class ConfluenceConnector(BaseConnector):
    """
    Connector for Confluence pages.

    Args:
        base_url: e.g. "https://mycompany.atlassian.net/wiki" (Cloud)
                  or "https://wiki.internal.example.com" (Server/DC).
        email / api_token: Cloud auth (preferred).
        username / password: Server/DC auth (alternative).
        space_key: optional space to scope listing (e.g. "KB").
        redis_client: optional Redis client for persistent sync manifests.
    """

    def __init__(
        self,
        base_url: str,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        space_key: Optional[str] = None,
        redis_client=None,
    ):
        super().__init__(redis_client=redis_client)
        self.base_url = base_url.rstrip("/")
        self.space_key = space_key
        self._session = requests.Session()
        if email and api_token:
            self._session.auth = (email, api_token)
        elif username and password:
            self._session.auth = (username, password)
        else:
            raise ValueError(
                "ConfluenceConnector requires (email, api_token) or "
                "(username, password)"
            )

    def connect(self) -> None:
        """
        Verify reachability and credentials against the space API.

        Raises:
            ConnectionError: unreachable or 401/403.
        """
        try:
            resp = self._session.get(
                f"{self.base_url}/rest/api/space",
                params={"limit": 1},
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectionError(
                f"Confluence connection failed for {self.base_url}: {exc}"
            ) from exc
        logger.debug("ConfluenceConnector connected to %s", self.base_url)

    def list_files(self) -> List[str]:
        """
        List all page URIs (optionally scoped to one space).

        Returns URIs of the form confluence://<base_url>/pages/<id>.
        """
        uris: List[str] = []
        start = 0
        limit = 100
        while True:
            params: Dict[str, object] = {
                "type": "page",
                "limit": limit,
                "start": start,
                "expand": "version",
            }
            if self.space_key:
                params["spaceKey"] = self.space_key
            resp = self._session.get(
                f"{self.base_url}/rest/api/content",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            for page in results:
                uris.append(self._page_uri(page["id"]))
            if len(results) < limit:
                break
            start += limit
        return uris

    def fetch(self, file_uri: str) -> bytes:
        """
        Fetch a page and return it as UTF-8 text bytes
        ("# <title>\\n\\n<body text>").

        Raises:
            ValueError: URI does not match this connector's base URL.
            FileNotFoundError: page does not exist.
        """
        page_id = self._parse_page_id(file_uri)
        resp = self._session.get(
            f"{self.base_url}/rest/api/content/{page_id}",
            params={"expand": "body.storage,version,space"},
            timeout=30,
        )
        if resp.status_code == 404:
            raise FileNotFoundError(f"Confluence page not found: {file_uri}")
        resp.raise_for_status()
        data = resp.json()
        title = data.get("title", "")
        body = data.get("body", {}).get("storage", {}).get("value", "")
        content = f"# {title}\n\n{_strip_html(body)}"
        return content.encode("utf-8")

    def watch(self) -> List[str]:
        """
        Return pages that are new or modified since the last watch() call.

        Version token: page version number (increments on every edit).
        """
        current: Dict[str, str] = {}
        start = 0
        limit = 100
        while True:
            params: Dict[str, object] = {
                "type": "page",
                "limit": limit,
                "start": start,
                "expand": "version",
            }
            if self.space_key:
                params["spaceKey"] = self.space_key
            resp = self._session.get(
                f"{self.base_url}/rest/api/content",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            for page in results:
                version = page.get("version", {}).get("number", "")
                current[self._page_uri(page["id"])] = str(version)
            if len(results) < limit:
                break
            start += limit
        manifest_key = f"connector:confluence:{self.base_url}:{self.space_key or 'all'}"
        return self._detect_changes(manifest_key, current)

    # -- internals -----------------------------------------------------------

    def _page_uri(self, page_id: str) -> str:
        return f"confluence://{self.base_url}/pages/{page_id}"

    def _parse_page_id(self, file_uri: str) -> str:
        expected = f"confluence://{self.base_url}/pages/"
        if not file_uri.startswith(expected):
            raise ValueError(
                f"URI '{file_uri}' does not belong to {self.base_url}"
            )
        return file_uri[len(expected):]

    def _validate_uri(self, file_uri: str) -> None:
        """Sanity-check a Confluence page URI before use."""
        self._parse_page_id(file_uri)
