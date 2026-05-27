import importlib.util
import sys
import time
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient


def load_adapter_app():
    service_dir = Path(__file__).resolve().parents[1]
    sys.modules.pop("db", None)
    sys.path.insert(0, str(service_dir))
    path = service_dir / "app.py"
    spec = importlib.util.spec_from_file_location("analysis_adapter_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules.pop("db", None)
    try:
        sys.path.remove(str(service_dir))
    except ValueError:
        pass
    return module


adapter_app = load_adapter_app()


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def _await_job(job_id: str, status: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with adapter_app.connect() as conn:
            row = conn.execute(
                "select status from analysis_jobs where job_id = ?", (job_id,)
            ).fetchone()
        if row and row["status"] == status:
            return
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach status={status} in {timeout_s}s")


def test_analyze_dry_run_completes_synchronously(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    posted: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/scorecards"):
            posted.update(kwargs.get("json") or {})
            return FakeResponse({"scorecard_id": "11111111-1111-4111-8111-111111111111"})
        raise AssertionError(f"unexpected POST to {url}")

    monkeypatch.setattr(adapter_app.httpx, "post", fake_post)
    response = TestClient(adapter_app.app).post(
        "/analyze", json={"actor": "user_1", "symbol": "BTCUSDT", "dry_run": True}
    )
    assert response.status_code == 200
    assert response.elapsed.total_seconds() < 0.1
    job_id = response.json()["job_id"]
    _await_job(job_id, status="succeeded")
    job = TestClient(adapter_app.app).get(f"/jobs/{job_id}").json()
    assert job["status"] == "succeeded"
    assert job["scorecard_id"] == "11111111-1111-4111-8111-111111111111"
    assert posted["actor"] == "user_1"
    assert posted["symbol"] == "BTCUSDT"
    assert posted["action"] == "buy"
    assert posted["source"] == "tradingagents"


def test_analyze_propagates_ta_decision_to_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    posted: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/analyze"):
            return FakeResponse({
                "ok": True,
                "decision": "SELL",
                "provider": "deepseek",
                "reports": {"market": "bearish", "news": "negative"},
            })
        if url.endswith("/scorecards"):
            posted.update(kwargs.get("json") or {})
            return FakeResponse({"scorecard_id": str(uuid4())})
        raise AssertionError(url)

    monkeypatch.setattr(adapter_app.httpx, "post", fake_post)
    job_id = TestClient(adapter_app.app).post(
        "/analyze", json={"actor": "u", "symbol": "ETHUSDT"}
    ).json()["job_id"]
    _await_job(job_id, status="succeeded")
    assert posted["action"] == "sell"
    assert posted["source"] == "tradingagents"
    assert posted["conviction"] == "0.7000"



def test_crypto_symbol_normalization_sent_to_tradingagents_bridge(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    seen: list[str] = []

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/analyze"):
            payload = kwargs.get("json") or {}
            assert isinstance(payload, dict)
            seen.append(str(payload["ticker"]))
            return FakeResponse({
                "ok": True,
                "decision": "HOLD",
                "provider": "deepseek",
                "reports": {"market": "rangebound", "news": "quiet"},
            })
        if url.endswith("/scorecards"):
            return FakeResponse({"scorecard_id": str(uuid4())})
        raise AssertionError(url)

    monkeypatch.setattr(adapter_app.httpx, "post", fake_post)
    client = TestClient(adapter_app.app)
    for symbol in ["BTCUSDT", "BTC/USDT", "bitcoin"]:
        job_id = client.post(
            "/analyze", json={"actor": "u", "symbol": symbol}
        ).json()["job_id"]
        _await_job(job_id, status="succeeded")

    assert seen == ["BTC-USD", "BTC-USD", "BTC-USD"]


def test_failed_unparseable_ta_response_is_saved_for_diagnosis(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    raw = {"ok": True, "decision": "PROBABLY", "final_trade_decision": "unclear"}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/analyze"):
            return FakeResponse(raw)
        raise AssertionError(url)

    monkeypatch.setattr(adapter_app.httpx, "post", fake_post)
    job_id = TestClient(adapter_app.app).post(
        "/analyze", json={"actor": "u", "symbol": "BTCUSDT"}
    ).json()["job_id"]
    _await_job(job_id, status="failed")

    with adapter_app.connect() as conn:
        row = conn.execute(
            "select raw_response_json from analysis_jobs where job_id = ?", (job_id,)
        ).fetchone()
    assert row is not None
    assert row["raw_response_json"] == adapter_app.json.dumps(raw, default=str)

def test_analyze_ta_failure_marks_job_failed(monkeypatch, tmp_path: Path) -> None:
    import httpx as _httpx

    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/analyze"):
            raise _httpx.HTTPError("connection refused")
        raise AssertionError(url)

    monkeypatch.setattr(adapter_app.httpx, "post", fake_post)
    job_id = TestClient(adapter_app.app).post(
        "/analyze", json={"actor": "u", "symbol": "BTCUSDT"}
    ).json()["job_id"]
    _await_job(job_id, status="failed")
    job = TestClient(adapter_app.app).get(f"/jobs/{job_id}").json()
    assert "connection refused" in job["error"]


def test_analyze_ta_returns_unparseable_decision(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/analyze"):
            return FakeResponse({"ok": True, "decision": "PROBABLY"})
        raise AssertionError(url)

    monkeypatch.setattr(adapter_app.httpx, "post", fake_post)
    job_id = TestClient(adapter_app.app).post(
        "/analyze", json={"actor": "u", "symbol": "BTCUSDT"}
    ).json()["job_id"]
    _await_job(job_id, status="failed")


def test_conviction_heuristic_buy_with_all_reports() -> None:
    score = adapter_app._derive_conviction(
        "buy",
        {"market": "x", "news": "y", "social": "z", "fundamentals": "w", "sentiment": "v"},
    )
    assert score == 0.90


def test_conviction_heuristic_hold_capped_low() -> None:
    assert adapter_app._derive_conviction("hold", {"market": "x"}) == 0.30


def test_get_jobs_filters_by_actor_and_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = TestClient(adapter_app.app)
    for actor, symbol in [("a", "BTCUSDT"), ("b", "ETHUSDT"), ("a", "SOLUSDT")]:
        with adapter_app.connect() as conn:
            conn.execute(
                "insert into analysis_jobs "
                "(job_id, actor, symbol, asset_type, requested_at, status, scorecard_id) "
                "values (?,?,?,?,?,?,?)",
                (
                    str(uuid4()),
                    actor,
                    symbol,
                    "crypto",
                    adapter_app._now().isoformat(),
                    "succeeded" if actor == "a" else "failed",
                    "sc",
                ),
            )
            conn.commit()
    response = client.get("/jobs", params={"actor": "a", "status": "succeeded"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 2
    assert {item["actor"] for item in body["items"]} == {"a"}


def test_get_job_unknown_returns_404(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    response = TestClient(adapter_app.app).get(
        "/jobs/99999999-9999-4999-8999-999999999999"
    )
    assert response.status_code == 404


def test_analyze_orchestrator_failure_marks_job_failed(monkeypatch, tmp_path: Path) -> None:
    import httpx as _httpx

    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith("/scorecards"):
            raise _httpx.HTTPError("orchestrator unreachable")
        raise AssertionError(url)

    monkeypatch.setattr(adapter_app.httpx, "post", fake_post)
    job_id = TestClient(adapter_app.app).post(
        "/analyze", json={"actor": "u", "symbol": "BTCUSDT", "dry_run": True}
    ).json()["job_id"]
    _await_job(job_id, status="failed")
