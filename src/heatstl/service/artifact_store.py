"""Pluggable artifact store for heatstl-produced binary outputs.

The engine publishes per-request artefacts (`.vtu`, `.xdmf`, `.h5`,
`.json`) via an :class:`ArtifactStore`. Two concrete backends ship:

- :class:`LocalFSStore` writes under ``HEATSTL_ARTIFACT_DIR`` and returns
  ``file://`` URIs. Default for dev / CI.
- :class:`GCSStore` writes to a GCS bucket and returns ``gs://<bucket>/<key>``
  URIs. Requires the ``google-cloud-storage`` dependency (already in the
  ``service`` extra).

Backend selection is driven by ``HEATSTL_ARTIFACT_STORE=local|gcs``
(default ``local``). The grid resolves the returned URIs through its
own asset proxy; the engine does not sign URLs itself.

The Protocol matches the one used by diagnostic-designer, so the
Analog-side wiring can be shared.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol
from urllib.parse import quote


class ArtifactStore(Protocol):
    """Storage backend for engine artefacts."""

    def put(self, key: str, data: bytes, *, content_type: str) -> str:
        """Write `data` under `key` and return a URI the grid can resolve."""

    def exists(self, key: str) -> str | None:
        """Return the URI if `key` is already present, else None."""


class ArtifactStoreConfigurationError(RuntimeError):
    """Raised when the selected artifact store is not deployable."""


# --------------------------------------------------------------------------- #
# Local filesystem
# --------------------------------------------------------------------------- #

class LocalFSStore:
    """Writes artefacts under ``root`` and returns ``file://`` URIs."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = os.environ.get("HEATSTL_ARTIFACT_DIR")
        if root is None:
            root = Path(tempfile.gettempdir()) / "heatstl-artifacts"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        # Reject paths that try to escape the root.
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"invalid artefact key {key!r}")
        return self.root / key

    def put(self, key: str, data: bytes, *, content_type: str) -> str:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.resolve().as_uri()

    def exists(self, key: str) -> str | None:
        path = self._path_for(key)
        return path.resolve().as_uri() if path.exists() else None


# --------------------------------------------------------------------------- #
# Google Cloud Storage
# --------------------------------------------------------------------------- #

class GCSStore:
    """Writes artefacts to a GCS bucket; returns ``gs://`` URIs.

    Bucket name comes from `bucket` arg or the ``HEATSTL_GCS_BUCKET``
    environment variable. Optional ``HEATSTL_GCS_PREFIX`` prepends a path
    segment to every key (useful for multi-tenant buckets).
    """

    def __init__(self, bucket: str | None = None, prefix: str | None = None) -> None:
        if bucket is None:
            bucket = os.environ.get("HEATSTL_GCS_BUCKET")
        if not bucket:
            raise ArtifactStoreConfigurationError(
                "GCSStore requires HEATSTL_GCS_BUCKET or an explicit `bucket` arg"
            )
        try:
            from google.cloud import storage  # type: ignore
        except ImportError as e:
            raise ArtifactStoreConfigurationError(
                "google-cloud-storage missing; install heatstl[service]"
            ) from e
        self._storage = storage
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.bucket_name = bucket
        self.prefix = prefix or os.environ.get("HEATSTL_GCS_PREFIX", "")

    def _full_key(self, key: str) -> str:
        # Normalise: strip leading slashes, reject "..".
        if ".." in Path(key).parts:
            raise ValueError(f"invalid artefact key {key!r}")
        clean = key.lstrip("/")
        return f"{self.prefix.rstrip('/')}/{clean}" if self.prefix else clean

    def put(self, key: str, data: bytes, *, content_type: str) -> str:
        full = self._full_key(key)
        blob = self.bucket.blob(full)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket_name}/{quote(full, safe='/')}"

    def exists(self, key: str) -> str | None:
        full = self._full_key(key)
        blob = self.bucket.blob(full)
        if blob.exists():
            return f"gs://{self.bucket_name}/{quote(full, safe='/')}"
        return None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def get_default_store() -> ArtifactStore:
    kind = os.environ.get("HEATSTL_ARTIFACT_STORE", "local").lower()
    if kind == "local":
        return LocalFSStore()
    if kind == "gcs":
        return GCSStore()
    raise ArtifactStoreConfigurationError(
        f"Unknown HEATSTL_ARTIFACT_STORE={kind!r}; expected 'local' or 'gcs'"
    )
