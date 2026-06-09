"""Tests for the unified SNAQCS client."""
from unittest.mock import MagicMock, patch

import pytest

from snaqcs import SnaqcsClient, AuthenticationError, Circuits, Decoder, DecoderResult, PropagationResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_resp(data, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


DECODE_PAYLOAD = {
    "error": "XIIII",
    "syndrome": [0, 0, 0, 1],
    "syndrome_string": "(0,0,0,1)",
    "syndrome_weight": 1,
    "recovery": "XIIII",
    "syndrome_in_table": True,
    "corrected_error": "IIIII",
    "corrected_error_phase": 0,
    "is_correctable": True,
    "correction_type": "perfect",
    "final_syndrome": [0, 0, 0, 0],
    "is_final_syndrome_trivial": True,
    "details": "Corrected",
}

PROPAGATE_PAYLOAD = {
    "is_correctable": True,
    "initial_error": "XIIIIII",
    "final_error": "XIIIXXX",
    "final_syndrome": {"X": [0, 0, 0], "Z": [1, 1, 1]},
    "final_correction": {"X": [], "Z": [0, 3, 4]},
    "code_parameters": {"n": 7, "k": 1, "d": 3, "t": 1},
    "intermediate_steps": [],
}


# ── Init / auth ───────────────────────────────────────────────────────────────

def test_no_key_means_no_auth_header(monkeypatch):
    monkeypatch.delenv("SNAQCS_API_KEY", raising=False)
    monkeypatch.delenv("SNAQCS_API_URL", raising=False)
    c = SnaqcsClient()
    assert "Authorization" not in c._session.headers


def test_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("SNAQCS_API_KEY", "snaqcs_fromenv")
    c = SnaqcsClient()
    assert c._session.headers["Authorization"] == "Bearer snaqcs_fromenv"


def test_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("SNAQCS_API_KEY", "snaqcs_env")
    c = SnaqcsClient(api_key="snaqcs_explicit")
    assert c._session.headers["Authorization"] == "Bearer snaqcs_explicit"


def test_reads_base_url_from_env(monkeypatch):
    monkeypatch.setenv("SNAQCS_API_URL", "https://snaqcs.point8.cloud")
    monkeypatch.delenv("SNAQCS_API_KEY", raising=False)
    c = SnaqcsClient()
    assert c.base_url == "https://snaqcs.point8.cloud"


def test_default_base_url_is_localhost(monkeypatch):
    monkeypatch.delenv("SNAQCS_API_URL", raising=False)
    c = SnaqcsClient()
    assert c.base_url == "http://localhost:6090"


def test_trailing_slash_stripped():
    c = SnaqcsClient(base_url="http://localhost:6090/")
    assert not c.base_url.endswith("/")


def test_401_raises_authentication_error():
    c = SnaqcsClient(api_key="snaqcs_bad")
    with patch.object(c._session, "post", return_value=_mock_resp({}, status=401)):
        with pytest.raises(AuthenticationError):
            c._post("/api/test", {})


# ── decode ────────────────────────────────────────────────────────────────────

def test_decode_returns_decoder_result():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._session, "post", return_value=_mock_resp(DECODE_PAYLOAD)):
        result = c.decode("XIIII", stabilizers=["+XZZXI"], num_qubits=5)
    assert isinstance(result, DecoderResult)
    assert result.is_correctable is True
    assert result.recovery == "XIIII"
    assert result.syndrome == (0, 0, 0, 1)


def test_decode_batch_returns_list():
    c = SnaqcsClient(api_key="snaqcs_x")
    batch_payload = {"results": [DECODE_PAYLOAD, DECODE_PAYLOAD]}
    with patch.object(c._session, "post", return_value=_mock_resp(batch_payload)):
        results = c.decode_batch(["XIIII", "XIIII"], stabilizers=["+XZZXI"])
    assert len(results) == 2
    assert all(isinstance(r, DecoderResult) for r in results)


# ── Decoder helper ────────────────────────────────────────────────────────────

def test_decoder_factory_returns_bound_decoder():
    c = SnaqcsClient(api_key="snaqcs_x")
    d = c.decoder(stabilizers=["+XZZXI", "+IXZZX"], num_qubits=5)
    assert isinstance(d, Decoder)
    assert d._client is c
    assert d.stabilizers == ["+XZZXI", "+IXZZX"]
    assert d.num_qubits == 5


def test_decoder_decode_delegates_to_client():
    c = SnaqcsClient(api_key="snaqcs_x")
    d = c.decoder(stabilizers=["+XZZXI"], num_qubits=5)
    with patch.object(c, "decode", return_value=MagicMock()) as mock:
        d.decode("XIIII")
    mock.assert_called_once_with(
        "XIIII",
        stabilizers=["+XZZXI"],
        num_qubits=5,
        logical_x=None,
        logical_z=None,
    )


def test_decoder_decode_batch_delegates_to_client():
    c = SnaqcsClient(api_key="snaqcs_x")
    d = c.decoder(stabilizers=["+XZZXI"], num_qubits=5)
    with patch.object(c, "decode_batch", return_value=[]) as mock:
        d.decode_batch(["XIIII"])
    mock.assert_called_once_with(
        ["XIIII"],
        stabilizers=["+XZZXI"],
        num_qubits=5,
        logical_x=None,
        logical_z=None,
    )


# ── propagate ─────────────────────────────────────────────────────────────────

def test_propagate_returns_propagation_result():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._session, "post", return_value=_mock_resp(PROPAGATE_PAYLOAD)):
        result = c.propagate(
            circuit={"qubits": 7, "layers": []},
            faults=[{"location": 5, "qubit": 0, "pauli": "X"}],
            code_config={"n": 7, "k": 1, "d": 3},
        )
    assert isinstance(result, PropagationResult)
    assert result.is_correctable is True
    assert result.initial_error == "XIIIIII"
    assert result.final_syndrome == {"X": [0, 0, 0], "Z": [1, 1, 1]}


# ── health ────────────────────────────────────────────────────────────────────

def test_health_uses_get():
    c = SnaqcsClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(c._session, "get", return_value=mock_resp) as mock_get:
        result = c.health()
    mock_get.assert_called_once()
    assert result == {"status": "ok"}


# ── wilson_ci ─────────────────────────────────────────────────────────────────

def test_wilson_ci_passes_correct_params():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._session, "post", return_value=_mock_resp({"ci_lower": 0.1, "ci_upper": 0.3, "ci_z": 1.96, "proportion": 0.1})) as mock_post:
        result = c.wilson_ci(num_success=10, num_total=100, ci_z=1.96)
    body = mock_post.call_args[1]["json"]
    assert body["num_success"] == 10
    assert body["num_total"] == 100
    assert body["ci_z"] == 1.96
    assert result["ci_lower"] == 0.1


# ── validate_noise_profile ────────────────────────────────────────────────────

def test_validate_noise_profile_posts_profile_body():
    c = SnaqcsClient(api_key="snaqcs_x")
    profile = {"1q_gate": {"rate": 1e-3}, "2q_gate": {"rate": 5e-3}}
    with patch.object(c._session, "post", return_value=_mock_resp({"depolarizing_config": {}, "correlated_config": None})) as mock_post:
        c.validate_noise_profile(profile)
    assert mock_post.call_args[1]["json"] == profile


# ── Circuits sub-client ───────────────────────────────────────────────────────

def test_circuits_property_returns_circuits_instance():
    c = SnaqcsClient(api_key="snaqcs_x")
    assert isinstance(c.circuits, Circuits)


def test_circuits_property_is_cached():
    c = SnaqcsClient(api_key="snaqcs_x")
    assert c.circuits is c.circuits


def test_circuits_list_uses_get():
    c = SnaqcsClient(api_key="snaqcs_x")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"name": "my_circ"}]
    mock_resp.raise_for_status = MagicMock()
    with patch.object(c._session, "get", return_value=mock_resp):
        entries = c.circuits.list()
    assert entries == [{"name": "my_circ"}]


def test_circuits_get_returns_none_on_404():
    c = SnaqcsClient(api_key="snaqcs_x")
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    with patch.object(c._session, "get", return_value=mock_resp):
        result = c.circuits.get("nonexistent")
    assert result is None


def test_circuits_save_posts_circuit():
    c = SnaqcsClient(api_key="snaqcs_x")
    circuit = {"name": "test", "qubits": 3, "layers": []}
    with patch.object(c._session, "post", return_value=_mock_resp({"name": "test"})) as mock_post:
        c.circuits.save(circuit)
    assert mock_post.call_args[1]["json"] == circuit


def test_circuits_delete_calls_delete_method():
    c = SnaqcsClient(api_key="snaqcs_x")
    mock_resp = MagicMock()
    mock_resp.status_code = 204
    mock_resp.raise_for_status = MagicMock()
    with patch.object(c._session, "delete", return_value=mock_resp) as mock_del:
        c.circuits.delete("my_circ")
    assert "/api/circuits/my_circ" in mock_del.call_args[0][0]


def test_circuits_import_qasm_posts_body():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._session, "post", return_value=_mock_resp({"name": "imported"})) as mock_post:
        c.circuits.import_qasm("OPENQASM 2.0;", name="imported")
    body = mock_post.call_args[1]["json"]
    assert body["qasm"] == "OPENQASM 2.0;"
    assert body["name"] == "imported"
