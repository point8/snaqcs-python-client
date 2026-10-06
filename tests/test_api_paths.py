"""Every URL the client calls must exist on the server.

The unit tests assert URLs for only a subset of call sites; this covers all of
them. ``SERVER_PATHS`` is a frozen copy of the server's route inventory
(snaqcs ``backend/tests/test_route_inventory.py``, the ``/api/v1`` surface of
2026-10-06), paths only, with every path parameter normalised to ``{param}``.
It is copied rather than imported so the client stays self-contained;
``test_integration.py`` repeats the check against a live ``/openapi.json``.
"""

import re
from pathlib import Path

SNAQCS_PY = Path(__file__).resolve().parents[1] / "snaqcs.py"

SERVER_PATHS = {
    "/api/v1/admin/library/gc",
    "/api/v1/admin/library/gc/purge",
    "/api/v1/analysis/circuit/enumerate_faults",
    "/api/v1/analysis/circuit/enumerate_pauli_faults",
    "/api/v1/analysis/circuit/propagate",
    "/api/v1/analysis/circuit/propagate_multi",
    "/api/v1/analysis/circuit/valid_fault_ticks",
    "/api/v1/analysis/protocol/hydrate",
    "/api/v1/analysis/protocol/propagate",
    "/api/v1/analysis/protocol/replay",
    "/api/v1/analysis/protocol/replayability",
    "/api/v1/analysis/wilson_ci",
    "/api/v1/circuits",
    "/api/v1/circuits/id/{param}",
    "/api/v1/circuits/id/{param}/versions/{param}",
    "/api/v1/circuits/import/qasm",
    "/api/v1/circuits/{param}",
    "/api/v1/circuits/{param}/export/json",
    "/api/v1/circuits/{param}/export/qasm",
    "/api/v1/circuits/{param}/owner",
    "/api/v1/circuits/{param}/visibility",
    "/api/v1/decoder/decode",
    "/api/v1/decoder/decode_batch",
    "/api/v1/decoder/info",
    "/api/v1/decoder/syndrome",
    "/api/v1/health",
    "/api/v1/import/braket",
    "/api/v1/import/qasm2",
    "/api/v1/import/qasm3",
    "/api/v1/import/stim",
    "/api/v1/me",
    "/api/v1/me/keys",
    "/api/v1/me/keys/{param}",
    "/api/v1/noise_models",
    "/api/v1/noise_models/id/{param}",
    "/api/v1/noise_models/id/{param}/versions",
    "/api/v1/noise_models/id/{param}/versions/{param}",
    "/api/v1/noise_models/profile/validate",
    "/api/v1/noise_models/{param}",
    "/api/v1/noise_models/{param}/owner",
    "/api/v1/noise_models/{param}/visibility",
    "/api/v1/protocols",
    "/api/v1/protocols/id/{param}",
    "/api/v1/protocols/{param}",
    "/api/v1/protocols/{param}/hydrated",
    "/api/v1/protocols/{param}/owner",
    "/api/v1/protocols/{param}/visibility",
    "/api/v1/sampling/circuit/direct",
    "/api/v1/sampling/jobs",
    "/api/v1/sampling/jobs/bulk_delete",
    "/api/v1/sampling/jobs/stream/me",
    "/api/v1/sampling/jobs/{param}",
    "/api/v1/sampling/jobs/{param}/cancel",
    "/api/v1/sampling/jobs/{param}/checkpoint",
    "/api/v1/sampling/jobs/{param}/stream",
    "/api/v1/sampling/protocol/direct",
    "/api/v1/sampling/protocol/subset",
    "/api/v1/sampling/workers",
    "/api/v1/sampling/workers/{param}",
    "/api/v1/version",
    "/api/v1/visualization/circuit",
    "/api/v1/visualization/protocol_graph",
    "/api/v1/visualization/typst",
}

# A quoted (optionally f-) string literal that starts with /api/.
_LITERAL = re.compile(r"""f?["'](/api/[^"']*)["']""")


def normalise(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{param}", path)


def client_paths() -> set[str]:
    return {normalise(m.group(1)) for m in _LITERAL.finditer(SNAQCS_PY.read_text())}


def test_client_literals_were_found():
    assert len(client_paths()) > 25


def test_every_client_path_is_a_server_route():
    assert client_paths() - SERVER_PATHS == set()
