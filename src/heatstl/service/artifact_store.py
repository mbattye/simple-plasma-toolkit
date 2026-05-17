"""Pluggable artifact store for heatstl-produced binary outputs.

The engine publishes per-request artefacts (`.vtu`, `.xdmf`, `.h5`,
`.json`, `.stl`) via an :class:`ArtifactStore`. Two concrete backends ship:

- :class:`LocalFSStore` writes under ``ENGINE_ARTIFACT_DIR`` and returns
  ``file://`` URIs. Default for dev / CI.
- :class:`S3Store` writes to an S3-compatible bucket (AWS S3 or Cloudflare
  R2) and returns ``s3://<bucket>/<key>`` URIs. Requires ``boto3``.

Backend selection is driven by ``ENGINE_ARTIFACT_STORE=local|s3`` (default
``local``). The env-var names match diagnostic-designer exactly, so the
same the-grid asset proxy resolves URIs from either engine.

For Cloudflare R2 deployments, set:
    ENGINE_ARTIFACT_STORE=s3
    ENGINE_S3_BUCKET=analog-twins
    ENGINE_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
    ENGINE_S3_ACCESS_KEY_ID=<R2 token access key>
    ENGINE_S3_SECRET_ACCESS_KEY=<R2 token secret>
    ENGINE_S3_REGION=auto       # R2 ignores this; "auto" is the convention
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Protocol


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
            root = os.environ.get("ENGINE_ARTIFACT_DIR") or os.environ.get(
                "HEATSTL_ARTIFACT_DIR"
            )
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
# S3-compatible (AWS S3 or Cloudflare R2)
# --------------------------------------------------------------------------- #

class S3Store:
    """S3-compatible :class:`ArtifactStore`.

    Reads ``ENGINE_S3_BUCKET``, ``ENGINE_S3_REGION``, ``ENGINE_S3_ENDPOINT``.
    Custom endpoints such as Cloudflare R2 require ``ENGINE_S3_ACCESS_KEY_ID``
    / ``ENGINE_S3_SECRET_ACCESS_KEY`` (plus optional
    ``ENGINE_S3_SESSION_TOKEN``). For AWS S3, the standard AWS credential
    chain is also supported. ``boto3`` is imported lazily so it remains an
    optional dependency.

    Mirrors diagnostic-designer's S3Store exactly — same env-var names, same
    URI format — so the-grid asset proxy can resolve both engines' outputs
    through a single code path.
    """

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
    ) -> None:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "S3Store requires boto3. Install with: pip install 'heatstl[service]'"
            ) from exc

        self.bucket = bucket or os.environ.get("ENGINE_S3_BUCKET")
        if not self.bucket:
            raise ArtifactStoreConfigurationError(
                "ENGINE_S3_BUCKET must be set for S3Store"
            )
        self.region = region or os.environ.get("ENGINE_S3_REGION")
        self.endpoint_url = endpoint_url or os.environ.get("ENGINE_S3_ENDPOINT")
        self.access_key_id = (
            access_key_id
            or os.environ.get("ENGINE_S3_ACCESS_KEY_ID")
            or os.environ.get("AWS_ACCESS_KEY_ID")
        )
        self.secret_access_key = (
            secret_access_key
            or os.environ.get("ENGINE_S3_SECRET_ACCESS_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        self.session_token = (
            session_token
            or os.environ.get("ENGINE_S3_SESSION_TOKEN")
            or os.environ.get("AWS_SESSION_TOKEN")
        )

        if self.endpoint_url and not (self.access_key_id and self.secret_access_key):
            raise ArtifactStoreConfigurationError(
                "ENGINE_ARTIFACT_STORE=s3 with ENGINE_S3_ENDPOINT requires credentials. "
                "Set ENGINE_S3_ACCESS_KEY_ID and ENGINE_S3_SECRET_ACCESS_KEY "
                "(or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)."
            )

        client_kwargs: dict = {
            "region_name": self.region,
            "endpoint_url": self.endpoint_url,
        }
        if self.access_key_id and self.secret_access_key:
            client_kwargs["aws_access_key_id"] = self.access_key_id
            client_kwargs["aws_secret_access_key"] = self.secret_access_key
        if self.session_token:
            client_kwargs["aws_session_token"] = self.session_token

        self._client = boto3.client("s3", **client_kwargs)

    def _uri_for(self, key: str) -> str:
        return f"s3://{self.bucket}/{key.lstrip('/')}"

    def put(self, key: str, data: bytes, *, content_type: str) -> str:
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key.lstrip("/"),
                Body=data,
                ContentType=content_type,
            )
        except Exception as exc:
            self._raise_if_missing_credentials(exc)
            raise
        return self._uri_for(key)

    def exists(self, key: str) -> str | None:
        from botocore.exceptions import ClientError  # type: ignore[import-not-found]

        try:
            self._client.head_object(Bucket=self.bucket, Key=key.lstrip("/"))
        except Exception as exc:
            self._raise_if_missing_credentials(exc)
            if not isinstance(exc, ClientError):
                raise
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

        return self._uri_for(key)

    @staticmethod
    def _raise_if_missing_credentials(exc: Exception) -> None:
        try:
            from botocore.exceptions import NoCredentialsError  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover
            return

        if isinstance(exc, NoCredentialsError):
            raise ArtifactStoreConfigurationError(
                "S3 artifact store credentials are missing. Set "
                "ENGINE_S3_ACCESS_KEY_ID and ENGINE_S3_SECRET_ACCESS_KEY "
                "(or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)."
            ) from exc


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_DEFAULT_STORE_LOCK = threading.Lock()
_DEFAULT_STORE: ArtifactStore | None = None


def get_default_store() -> ArtifactStore:
    """Return the process-wide default :class:`ArtifactStore`.

    Selection is controlled by ``ENGINE_ARTIFACT_STORE`` (``local`` | ``s3``).
    """
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            # Honour legacy HEATSTL_ARTIFACT_STORE for back-compat during the
            # GCS→R2 migration; ENGINE_ARTIFACT_STORE wins.
            kind = os.environ.get(
                "ENGINE_ARTIFACT_STORE",
                os.environ.get("HEATSTL_ARTIFACT_STORE", "local"),
            ).lower()
            if kind == "local":
                _DEFAULT_STORE = LocalFSStore()
            elif kind == "s3":
                _DEFAULT_STORE = S3Store()
            else:
                raise ArtifactStoreConfigurationError(
                    f"Unknown ENGINE_ARTIFACT_STORE={kind!r}; expected 'local' or 's3'"
                )
        return _DEFAULT_STORE


def reset_default_store() -> None:
    """Test helper: drop the cached default store so the next call rebuilds it."""
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        _DEFAULT_STORE = None
