from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from wemm_kaggle.api.auth import build_auth_dependency

TOKEN = "t" * 40


def build_app():
    app = FastAPI()
    require = build_auth_dependency(TOKEN)

    @app.get("/protected")
    def protected(_: None = Depends(require)):
        return {"ok": True}

    return app


def client():
    return TestClient(build_app())


def test_missing_auth_header_is_401_with_www_authenticate():
    resp = client().get("/protected")
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_wrong_bearer_is_401():
    resp = client().get("/protected", headers={"Authorization": "Bearer not-the-token"})
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_correct_bearer_is_200():
    resp = client().get("/protected", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200


def test_malformed_header_without_bearer_scheme_is_401():
    resp = client().get("/protected", headers={"Authorization": "Basic abc123"})
    assert resp.status_code == 401