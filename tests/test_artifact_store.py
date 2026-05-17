"""Artifact store unit tests — LocalFSStore, S3Store config, factory.

S3Store's `put`/`exists` paths exercise the boto3 client via a fake to
avoid real network calls. The objective here is to confirm the env-var
plumbing matches diagnostic-designer (and therefore the-grid integration
contract), not to retest boto3 itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from heatstl.service.artifact_store import (
    ArtifactStoreConfigurationError,
    LocalFSStore,
    get_default_store,
    reset_default_store,
)


# --------------------------------------------------------------------------- #
# LocalFSStore
# --------------------------------------------------------------------------- #

def test_local_store_put_and_exists(tmp_path):
    store = LocalFSStore(root=tmp_path)
    uri = store.put("foo/bar.txt", b"hello", content_type="text/plain")
    assert uri.startswith("file://")
    assert uri.endswith("foo/bar.txt")
    assert (tmp_path / "foo" / "bar.txt").read_bytes() == b"hello"
    assert store.exists("foo/bar.txt") == uri
    assert store.exists("nope") is None


def test_local_store_rejects_path_traversal(tmp_path):
    store = LocalFSStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.put("../escape.txt", b"x", content_type="text/plain")
    with pytest.raises(ValueError):
        store.put("/abs/path.txt", b"x", content_type="text/plain")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def test_factory_local(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGINE_ARTIFACT_STORE", "local")
    monkeypatch.setenv("ENGINE_ARTIFACT_DIR", str(tmp_path))
    reset_default_store()
    s = get_default_store()
    assert isinstance(s, LocalFSStore)


def test_factory_rejects_unknown(monkeypatch):
    monkeypatch.setenv("ENGINE_ARTIFACT_STORE", "bogus")
    reset_default_store()
    with pytest.raises(ArtifactStoreConfigurationError):
        get_default_store()


def test_factory_falls_back_to_legacy_var(monkeypatch, tmp_path):
    monkeypatch.delenv("ENGINE_ARTIFACT_STORE", raising=False)
    monkeypatch.setenv("HEATSTL_ARTIFACT_STORE", "local")
    monkeypatch.setenv("ENGINE_ARTIFACT_DIR", str(tmp_path))
    reset_default_store()
    s = get_default_store()
    assert isinstance(s, LocalFSStore)


# --------------------------------------------------------------------------- #
# S3Store config — exercises the env-var contract without hitting R2
# --------------------------------------------------------------------------- #

@pytest.fixture
def s3_env(monkeypatch):
    """Set the full ENGINE_S3_* env-var set (R2 style) and return the values."""
    vals = {
        "ENGINE_S3_BUCKET": "analog-twins",
        "ENGINE_S3_ENDPOINT": "https://example.r2.cloudflarestorage.com",
        "ENGINE_S3_ACCESS_KEY_ID": "AKIA-test",
        "ENGINE_S3_SECRET_ACCESS_KEY": "secret-test",
        "ENGINE_S3_REGION": "auto",
    }
    for k, v in vals.items():
        monkeypatch.setenv(k, v)
    return vals


def test_s3_store_init_reads_env(s3_env, monkeypatch):
    pytest.importorskip("boto3")
    from heatstl.service.artifact_store import S3Store

    captured = {}

    class _FakeClient:
        def put_object(self, **kw):
            captured["put"] = kw

        def head_object(self, **kw):
            captured["head"] = kw

    def _fake_boto3_client(service, **kwargs):
        assert service == "s3"
        captured["init"] = kwargs
        return _FakeClient()

    import boto3
    monkeypatch.setattr(boto3, "client", _fake_boto3_client)

    store = S3Store()
    assert store.bucket == s3_env["ENGINE_S3_BUCKET"]
    assert store.endpoint_url == s3_env["ENGINE_S3_ENDPOINT"]
    init = captured["init"]
    assert init["endpoint_url"] == s3_env["ENGINE_S3_ENDPOINT"]
    assert init["region_name"] == "auto"
    assert init["aws_access_key_id"] == s3_env["ENGINE_S3_ACCESS_KEY_ID"]
    assert init["aws_secret_access_key"] == s3_env["ENGINE_S3_SECRET_ACCESS_KEY"]

    uri = store.put("heatstl/run-1/result.xdmf", b"<xdmf/>", content_type="application/xml")
    assert uri == f"s3://{s3_env['ENGINE_S3_BUCKET']}/heatstl/run-1/result.xdmf"
    assert captured["put"]["Bucket"] == s3_env["ENGINE_S3_BUCKET"]
    assert captured["put"]["Key"] == "heatstl/run-1/result.xdmf"


def test_s3_store_requires_bucket(monkeypatch):
    pytest.importorskip("boto3")
    monkeypatch.delenv("ENGINE_S3_BUCKET", raising=False)
    from heatstl.service.artifact_store import S3Store
    with pytest.raises(ArtifactStoreConfigurationError):
        S3Store()


def test_s3_store_requires_creds_with_custom_endpoint(monkeypatch):
    pytest.importorskip("boto3")
    monkeypatch.setenv("ENGINE_S3_BUCKET", "b")
    monkeypatch.setenv("ENGINE_S3_ENDPOINT", "https://example.r2.cloudflarestorage.com")
    monkeypatch.delenv("ENGINE_S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ENGINE_S3_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    from heatstl.service.artifact_store import S3Store
    with pytest.raises(ArtifactStoreConfigurationError):
        S3Store()
