
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "deploy" / "docker-compose.yml"


def test_orchestrator_and_analysis_adapter_bind_host_local_only() -> None:
    text = COMPOSE.read_text()
    assert '"8080:8080"' not in text
    assert '"127.0.0.1:${ORCHESTRATOR_HOST_PORT:-18081}:8080"' in text
    assert '"127.0.0.1:${ANALYSIS_ADAPTER_HOST_PORT:-18085}:8085"' in text
