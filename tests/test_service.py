"""FastAPI service surface tests.

Two layers:

- Trivial endpoints (/health, /presets, auth) — exercised through FastAPI's
  TestClient.
- Heavy solve endpoints (/solve/steady, /solve/transient) — called by
  invoking the endpoint functions directly. Going through TestClient hangs
  on macOS when gmsh + scikit-fem run inside starlette's threadpool, but
  the underlying logic is identical (Pydantic validates either way).
"""

from __future__ import annotations

import pathlib
from urllib.parse import urlparse

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


# Stub httpx.Client so the engine's STL fetch hits a local file instead of
# the network. We replace `httpx.Client` after `starlette.testclient` has
# already captured its own reference, so the TestClient itself is
# unaffected.
@pytest.fixture
def fake_stl_url(monkeypatch):
    repo = pathlib.Path(__file__).resolve().parents[1]
    src_stl = repo / "examples" / "heat_shield_tile.stl"
    sentinel_url = "http://test.local/heat_shield_tile.stl"

    class _FakeResp:
        def __init__(self, body: bytes):
            self.content = body
            self.status_code = 200

        def raise_for_status(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, *args, **kwargs):
            return _FakeResp(src_stl.read_bytes())

    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    return sentinel_url


@pytest.fixture
def tmp_artifact_dir(monkeypatch, tmp_path):
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    monkeypatch.setenv("HEATSTL_ARTIFACT_STORE", "local")
    monkeypatch.setenv("HEATSTL_ARTIFACT_DIR", str(art_dir))
    from heatstl.service import app as service_app
    service_app._store = None
    return art_dir


@pytest.fixture
def client():
    from heatstl.service.app import app
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Trivial endpoints (TestClient)
# --------------------------------------------------------------------------- #

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_presets(client):
    r = client.get("/presets")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()]
    assert "starship" in names
    assert "starship-flip-conservative" in names


def test_steady_requires_token_when_set(client, fake_stl_url, tmp_artifact_dir, monkeypatch):
    monkeypatch.setenv("ENGINE_SECRET", "shh")
    from heatstl.service import app as service_app
    service_app.ENGINE_SECRET = "shh"
    try:
        r = client.post(
            "/solve/steady",
            json={
                "stl_url": fake_stl_url,
                "instance_id": "auth-test",
                "q0": 5e6,
            },
        )
        assert r.status_code == 401, r.text
    finally:
        service_app.ENGINE_SECRET = ""


# --------------------------------------------------------------------------- #
# Heavy solve endpoints (direct call, bypassing TestClient threadpool)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_steady_solve_v1_tile(fake_stl_url, tmp_artifact_dir):
    from heatstl.service.app import SteadyRequest, solve_steady_endpoint

    req = SteadyRequest(
        stl_url=fake_stl_url,
        instance_id="svc-steady",
        q0=5e6,
        angle_deg=0.0,
        azimuth_deg=0.0,
        k=150.0,
        bc_unheated="dirichlet",
        T_cool=300.0,
        unit="mm",
    )
    resp = solve_steady_endpoint(req, x_engine_token=None)

    assert resp.instance_id == "svc-steady"
    assert 800.0 < resp.peak_T < 1300.0
    assert resp.result_uri.startswith("file://")
    assert resp.report_uri.startswith("file://")

    for uri in (resp.result_uri, resp.report_uri):
        path = pathlib.Path(urlparse(uri).path)
        assert path.exists() and path.stat().st_size > 0


@pytest.mark.slow
def test_transient_solve_short_flip(fake_stl_url, tmp_artifact_dir):
    from heatstl.service.app import TransientRequest, solve_transient_endpoint

    req = TransientRequest(
        stl_url=fake_stl_url,
        instance_id="svc-flip",
        preset="starship-flip-conservative",
        q0=1e5,
        neighbors="none",
        unit="mm",
        duration=100.0,
        n_steps=10,
    )
    resp = solve_transient_endpoint(req, x_engine_token=None)

    assert resp.instance_id == "svc-flip"
    assert resp.n_steps == 10
    assert resp.result_uri.endswith(".xdmf")
    assert resp.h5_uri.endswith(".h5")
    assert resp.arrow_uri.endswith(".xdmf")
    assert resp.arrow_h5_uri.endswith(".h5")
    assert 300.0 < resp.peak_T < 1500.0

    for uri in (resp.result_uri, resp.h5_uri, resp.arrow_uri,
                resp.arrow_h5_uri, resp.report_uri):
        path = pathlib.Path(urlparse(uri).path)
        assert path.exists() and path.stat().st_size > 0, uri
