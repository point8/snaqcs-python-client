"""Tests for the unified SNAQCS client."""
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from snaqcs import (
    SnaqcsClient,
    AuthenticationError,
    Circuits,
    Decoder,
    DecoderResult,
    PropagationResult,
    Jobs,
    SamplerJob,
    ServerUnavailableError,
    JobError,
    JobFailedError,
    JobCancelledError,
    JobNotCompletedError,
)


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
    "initial_error": "XIIIIII",
    "final_error": "XIIIXXX",
    "has_superposition": False,
    "intermediate_steps": [],
}


# ── Init / auth ───────────────────────────────────────────────────────────────

def test_no_key_means_no_auth_header(monkeypatch):
    monkeypatch.delenv("SNAQCS_API_KEY", raising=False)
    monkeypatch.delenv("SNAQCS_API_URL", raising=False)
    c = SnaqcsClient()
    assert "Authorization" not in c._transport._session.headers


def test_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("SNAQCS_API_KEY", "snaqcs_fromenv")
    c = SnaqcsClient()
    assert c._transport._session.headers["Authorization"] == "Bearer snaqcs_fromenv"


def test_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("SNAQCS_API_KEY", "snaqcs_env")
    c = SnaqcsClient(api_key="snaqcs_explicit")
    assert c._transport._session.headers["Authorization"] == "Bearer snaqcs_explicit"


def test_reads_base_url_from_env(monkeypatch):
    monkeypatch.setenv("SNAQCS_API_URL", "https://snaqcs.point8.cloud")
    # A remote base URL without a key would fail fast in the constructor,
    # so provide one — the assertion is about the URL, not auth.
    monkeypatch.setenv("SNAQCS_API_KEY", "snaqcs_x")
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
    with patch.object(c._transport._session, "post", return_value=_mock_resp({}, status=401)):
        with pytest.raises(AuthenticationError):
            c._post("/api/test", {})


# ── decode ────────────────────────────────────────────────────────────────────

def test_decode_returns_decoder_result():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._transport._session, "post", return_value=_mock_resp(DECODE_PAYLOAD)):
        result = c.decode("XIIII", stabilizers=["+XZZXI"], num_qubits=5)
    assert isinstance(result, DecoderResult)
    assert result.is_correctable is True
    assert result.recovery == "XIIII"
    assert result.syndrome == (0, 0, 0, 1)


def test_decode_batch_returns_list():
    c = SnaqcsClient(api_key="snaqcs_x")
    batch_payload = {"results": [DECODE_PAYLOAD, DECODE_PAYLOAD]}
    with patch.object(c._transport._session, "post", return_value=_mock_resp(batch_payload)):
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
    with patch.object(c._transport._session, "post", return_value=_mock_resp(PROPAGATE_PAYLOAD)):
        result = c.propagate(
            circuit={"qubits": 7, "layers": []},
            faults=[{"location": 5, "qubit": 0, "pauli": "X"}],
        )
    assert isinstance(result, PropagationResult)
    assert result.initial_error == "XIIIIII"
    assert result.final_error == "XIIIXXX"
    assert result.has_superposition is False


# ── health ────────────────────────────────────────────────────────────────────

def test_health_uses_get():
    c = SnaqcsClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(c._transport._session, "get", return_value=mock_resp) as mock_get:
        result = c.health()
    mock_get.assert_called_once()
    assert result == {"status": "ok"}


# ── wilson_ci ─────────────────────────────────────────────────────────────────

def test_wilson_ci_passes_correct_params():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._transport._session, "post", return_value=_mock_resp({"ci_lower": 0.1, "ci_upper": 0.3, "ci_z": 1.96, "proportion": 0.1})) as mock_post:
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
    with patch.object(c._transport._session, "post", return_value=_mock_resp({"depolarizing_config": {}, "correlated_config": None})) as mock_post:
        c.validate_noise_profile(profile)
    assert mock_post.call_args[1]["json"] == profile


# ── sample ────────────────────────────────────────────────────────────────────

def test_sample_omits_decoder_fields_when_not_set():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._transport._session, "post", return_value=_mock_resp({"num_samples": 1})) as mock_post:
        c.sample(circuit={"qubits": 1, "layers": []}, check_functions={"a": "weight > 0"})
    body = mock_post.call_args[1]["json"]
    assert "propagation_backend" not in body
    assert "decoder_backend" not in body
    assert "decoder_config" not in body


def test_sample_passes_decoder_fields_when_set():
    c = SnaqcsClient(api_key="snaqcs_x")
    decoder_config = {"stabilizers": ["+ZZI", "+IZZ"], "num_qubits": 3}
    with patch.object(c._transport._session, "post", return_value=_mock_resp({"num_samples": 1})) as mock_post:
        c.sample(
            circuit={"qubits": 3, "layers": []},
            check_functions={"a": "weight > 0"},
            propagation_backend="numpy",
            decoder_backend="numpy",
            decoder_config=decoder_config,
        )
    body = mock_post.call_args[1]["json"]
    assert body["propagation_backend"] == "numpy"
    assert body["decoder_backend"] == "numpy"
    assert body["decoder_config"] == decoder_config


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
    with patch.object(c._transport._session, "get", return_value=mock_resp):
        entries = c.circuits.list()
    assert entries == [{"name": "my_circ"}]


def test_circuits_get_returns_none_on_404():
    c = SnaqcsClient(api_key="snaqcs_x")
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    with patch.object(c._transport._session, "get", return_value=mock_resp):
        result = c.circuits.get("nonexistent")
    assert result is None


def test_circuits_save_posts_circuit():
    c = SnaqcsClient(api_key="snaqcs_x")
    circuit = {"name": "test", "qubits": 3, "layers": []}
    with patch.object(c._transport._session, "post", return_value=_mock_resp({"name": "test"})) as mock_post:
        c.circuits.save(circuit)
    assert mock_post.call_args[1]["json"] == circuit


def test_circuits_delete_calls_delete_method():
    c = SnaqcsClient(api_key="snaqcs_x")
    mock_resp = MagicMock()
    mock_resp.status_code = 204
    mock_resp.raise_for_status = MagicMock()
    with patch.object(c._transport._session, "delete", return_value=mock_resp) as mock_del:
        c.circuits.delete("my_circ")
    assert "/api/circuits/my_circ" in mock_del.call_args[0][0]


def test_circuits_import_qasm_posts_body():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._transport._session, "post", return_value=_mock_resp({"name": "imported"})) as mock_post:
        c.circuits.import_qasm("OPENQASM 2.0;", name="imported")
    body = mock_post.call_args[1]["json"]
    assert body["qasm"] == "OPENQASM 2.0;"
    assert body["name"] == "imported"


def test_circuits_list_connection_error_raises_server_unavailable_error():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._transport._session, "get", side_effect=requests.exceptions.ConnectionError):
        with pytest.raises(ServerUnavailableError):
            c.circuits.list()


def test_circuits_get_connection_error_raises_server_unavailable_error():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._transport._session, "get", side_effect=requests.exceptions.ConnectionError):
        with pytest.raises(ServerUnavailableError):
            c.circuits.get("my_circ")


def test_circuits_delete_connection_error_raises_server_unavailable_error():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._transport._session, "delete", side_effect=requests.exceptions.ConnectionError):
        with pytest.raises(ServerUnavailableError):
            c.circuits.delete("my_circ")


def test_circuits_export_qasm_connection_error_raises_server_unavailable_error():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._transport._session, "get", side_effect=requests.exceptions.ConnectionError):
        with pytest.raises(ServerUnavailableError):
            c.circuits.export_qasm("my_circ")


# ── enumerate_pauli_faults ────────────────────────────────────────────────────

ENUMERATE_PAULI_PAYLOAD = {
    "total_faults": 3,
    "faults": [
        {"location": 0, "qubit": 0, "pauli": "X"},
        {"location": 0, "qubit": 0, "pauli": "Y"},
        {"location": 0, "qubit": 0, "pauli": "Z"},
    ],
}


def test_enumerate_pauli_faults_posts_to_correct_url():
    c = SnaqcsClient(api_key="snaqcs_x")
    circuit = {"qubits": 3, "layers": []}
    with patch.object(c._transport._session, "post", return_value=_mock_resp(ENUMERATE_PAULI_PAYLOAD)) as mock_post:
        result = c.enumerate_pauli_faults(circuit=circuit)
    url = mock_post.call_args[0][0]
    assert url.endswith("/api/enumerate_pauli_faults")
    assert result["total_faults"] == 3


def test_enumerate_pauli_faults_default_fault_types():
    c = SnaqcsClient(api_key="snaqcs_x")
    circuit = {"qubits": 3, "layers": []}
    with patch.object(c._transport._session, "post", return_value=_mock_resp(ENUMERATE_PAULI_PAYLOAD)) as mock_post:
        c.enumerate_pauli_faults(circuit=circuit)
    body = mock_post.call_args[1]["json"]
    assert body["fault_types"] == ["X", "Y", "Z"]


def test_enumerate_single_faults_alias_posts_to_correct_url():
    """Backward-compat alias must still reach /api/enumerate_pauli_faults."""
    c = SnaqcsClient(api_key="snaqcs_x")
    circuit = {"qubits": 3, "layers": []}
    with patch.object(c._transport._session, "post", return_value=_mock_resp(ENUMERATE_PAULI_PAYLOAD)) as mock_post:
        result = c.enumerate_single_faults(circuit=circuit)
    url = mock_post.call_args[0][0]
    assert url.endswith("/api/enumerate_pauli_faults")
    assert result["total_faults"] == 3


# ── Jobs sub-client ───────────────────────────────────────────────────────────

def _job_snapshot(status="queued", **overrides):
    snap = {
        "id": "11111111-1111-1111-1111-111111111111",
        "kind": "direct_sampler",
        "status": status,
        "request": {},
        "result": None,
        "error": None,
        "progress": None,
    }
    snap.update(overrides)
    return snap


def test_jobs_property_returns_jobs_instance():
    c = SnaqcsClient(api_key="snaqcs_x")
    assert isinstance(c.jobs, Jobs)


def test_jobs_property_is_cached():
    c = SnaqcsClient(api_key="snaqcs_x")
    assert c.jobs is c.jobs


def test_submit_direct_sampler_posts_kind_and_request_then_fetches_snapshot():
    # kind="direct_sampler" jobs wrap a protocol-graph request (the same
    # shape sample_protocol() posts to /api/protocol/direct_sampler), not
    # the single-circuit sample() shape.
    c = SnaqcsClient(api_key="snaqcs_x")
    request = {"config": {"circuits": {}, "edges": []}, "num_samples": 20}
    with patch.object(c._transport._session, "post", return_value=_mock_resp({"job_id": "abc-123"})) as mock_post, \
         patch.object(c._transport._session, "get", return_value=_mock_resp(_job_snapshot())) as mock_get:
        job = c.jobs.submit_direct_sampler(request)
    assert mock_post.call_args[1]["json"] == {"kind": "direct_sampler", "request": request}
    assert mock_get.call_args[0][0].endswith("/api/sampler/jobs/abc-123")
    assert isinstance(job, SamplerJob)
    assert job.status == "queued"


def test_submit_circuit_direct_sampler_posts_kind_and_request():
    # kind="circuit_direct_sampler" wraps the same shape sample() posts to
    # /api/direct_sampler — circuit + check_functions, optionally
    # decoder_backend/decoder_config.
    c = SnaqcsClient(api_key="snaqcs_x")
    request = {
        "circuit": {"qubits": 1, "layers": []},
        "check_functions": {"a": "weight > 0"},
        "decoder_backend": "stim",
        "decoder_config": {"stabilizers": ["+Z"], "num_qubits": 1},
    }
    with patch.object(c._transport._session, "post", return_value=_mock_resp({"job_id": "xyz-789"})) as mock_post, \
         patch.object(c._transport._session, "get", return_value=_mock_resp(_job_snapshot())) as mock_get:
        job = c.jobs.submit_circuit_direct_sampler(request)
    assert mock_post.call_args[1]["json"] == {"kind": "circuit_direct_sampler", "request": request}
    assert mock_get.call_args[0][0].endswith("/api/sampler/jobs/xyz-789")
    assert isinstance(job, SamplerJob)


def test_jobs_get_returns_sampler_job():
    c = SnaqcsClient(api_key="snaqcs_x")
    with patch.object(c._transport._session, "get", return_value=_mock_resp(_job_snapshot(status="running"))):
        job = c.jobs.get("11111111-1111-1111-1111-111111111111")
    assert isinstance(job, SamplerJob)
    assert job.status == "running"


def test_jobs_list_builds_query_params():
    c = SnaqcsClient(api_key="snaqcs_x")
    payload = {"items": [_job_snapshot(), _job_snapshot()], "limit": 50, "offset": 0}
    with patch.object(c._transport._session, "get", return_value=_mock_resp(payload)) as mock_get:
        jobs = c.jobs.list(status="queued", kind="direct_sampler", limit=10, offset=5)
    assert mock_get.call_args[1]["params"] == {
        "limit": 10, "offset": 5, "status": "queued", "kind": "direct_sampler",
    }
    assert len(jobs) == 2
    assert all(isinstance(j, SamplerJob) for j in jobs)


# ── SamplerJob ────────────────────────────────────────────────────────────────

def test_sampler_job_result_raises_when_not_completed():
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="running"))
    with pytest.raises(JobNotCompletedError):
        job.result


def test_sampler_job_result_returns_result_dict_when_completed():
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="completed", result={"num_samples": 10}))
    assert job.result == {"num_samples": 10}


def test_sampler_job_timestamps_parse_to_datetime():
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(
        status="completed",
        submitted_at="2026-07-16T08:00:00+00:00",
        started_at="2026-07-16T08:00:01+00:00",
        finished_at="2026-07-16T08:09:21+00:00",
    ))
    assert (job.finished_at - job.started_at).total_seconds() == 560.0
    assert job.submitted_at.year == 2026


def test_sampler_job_timestamps_none_when_absent():
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="queued"))
    assert job.submitted_at is None
    assert job.started_at is None
    assert job.finished_at is None


def test_sampler_job_refresh_refetches_snapshot():
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="queued"))
    with patch.object(c._transport._session, "get", return_value=_mock_resp(_job_snapshot(status="running"))):
        job.refresh()
    assert job.status == "running"


def test_sampler_job_wait_polls_until_completed(monkeypatch):
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="queued"))
    snapshots = iter([_job_snapshot(status="running"), _job_snapshot(status="completed", result={})])
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with patch.object(c._transport._session, "get", side_effect=lambda *a, **k: _mock_resp(next(snapshots))):
        result = job.wait(poll=0)
    assert result is job
    assert job.status == "completed"


def test_sampler_job_wait_raises_job_failed_error(monkeypatch):
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="running"))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with patch.object(
        c._transport._session, "get",
        return_value=_mock_resp(_job_snapshot(status="failed", error={"message": "boom"})),
    ):
        with pytest.raises(JobFailedError, match="boom"):
            job.wait(poll=0)


def test_sampler_job_wait_raises_job_cancelled_error(monkeypatch):
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="running"))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with patch.object(c._transport._session, "get", return_value=_mock_resp(_job_snapshot(status="cancelled"))):
        with pytest.raises(JobCancelledError):
            job.wait(poll=0)


def test_sampler_job_wait_raises_timeout_error(monkeypatch):
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="running"))
    times = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with patch.object(c._transport._session, "get", return_value=_mock_resp(_job_snapshot(status="running"))):
        with pytest.raises(TimeoutError):
            job.wait(timeout=5.0, poll=0)


def test_sampler_job_cancel_posts_reason_and_refreshes():
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="running"))
    with patch.object(c._transport._session, "post", return_value=_mock_resp({"cancel_requested": True})) as mock_post, \
         patch.object(c._transport._session, "get", return_value=_mock_resp(_job_snapshot(status="running"))):
        job.cancel(reason="no longer needed")
    assert mock_post.call_args[1]["json"] == {"reason": "no longer needed"}
    assert mock_post.call_args[0][0].endswith("/cancel")


def test_sampler_job_delete_calls_delete_endpoint():
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot())
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    with patch.object(c._transport._session, "delete", return_value=mock_resp) as mock_delete:
        job.delete()
    assert mock_delete.call_args[0][0].endswith(f"/api/sampler/jobs/{job.id}")


def test_sampler_job_delete_401_raises_authentication_error():
    c = SnaqcsClient(api_key="snaqcs_bad")
    job = SamplerJob(c, _job_snapshot())
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    with patch.object(c._transport._session, "delete", return_value=mock_resp):
        with pytest.raises(AuthenticationError):
            job.delete()


def test_sampler_job_delete_connection_error_raises_server_unavailable_error():
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot())
    with patch.object(c._transport._session, "delete", side_effect=requests.exceptions.ConnectionError):
        with pytest.raises(ServerUnavailableError):
            job.delete()


def test_sampler_job_stream_yields_sse_events_and_updates_snapshot():
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="queued"))

    lines = [
        'data: {"id": "11111111-1111-1111-1111-111111111111", "status": "running"}',
        'data: {"id": "11111111-1111-1111-1111-111111111111", "status": "completed", "result": {}}',
    ]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.__enter__ = lambda self: mock_resp
    mock_resp.__exit__ = lambda self, *a: None

    with patch.object(c._transport._session, "get", return_value=mock_resp):
        events = list(job.stream())

    assert [e["status"] for e in events] == ["running", "completed"]
    assert job.status == "completed"


def test_sampler_job_stream_falls_back_to_polling_on_connection_error():
    import requests

    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="queued"))

    with patch.object(c._transport._session, "get") as mock_get:
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("stream unavailable"),
            _mock_resp(_job_snapshot(status="completed", result={})),
        ]
        with patch("time.sleep", return_value=None):
            events = list(job.stream())

    assert events[-1]["status"] == "completed"


def test_sampler_job_stream_401_raises_authentication_error_not_polling_fallback():
    """A real auth failure must not be misclassified as a transient network
    drop — it should raise immediately, not silently start polling for one
    interval before eventually surfacing the same error."""
    c = SnaqcsClient(api_key="snaqcs_bad")
    job = SamplerJob(c, _job_snapshot(status="queued"))

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.__enter__ = lambda self: mock_resp
    mock_resp.__exit__ = lambda self, *a: None

    with patch.object(c._transport._session, "get", return_value=mock_resp) as mock_get:
        with pytest.raises(AuthenticationError):
            list(job.stream())
    # Only the one SSE connection attempt — no fallback polling GET fired.
    assert mock_get.call_count == 1


def test_sampler_job_stream_error_sentinel_raises_job_error():
    """The backend emits {"error": "not_found"} once when a job is
    soft-deleted mid-stream (see _job_stream_events's docstring). Silently
    adopting that shapeless dict as self._snap would break .status/.id on
    the next access with a bare KeyError instead of a clear signal."""
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="running"))

    lines = [
        'data: {"id": "11111111-1111-1111-1111-111111111111", "status": "running"}',
        'data: {"error": "not_found"}',
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.__enter__ = lambda self: mock_resp
    mock_resp.__exit__ = lambda self, *a: None

    with patch.object(c._transport._session, "get", return_value=mock_resp):
        with pytest.raises(JobError, match="not_found"):
            list(job.stream())
    # The last *good* snapshot must survive — not clobbered by the error dict.
    assert job.status == "running"


def test_sampler_job_stream_handles_blank_line_sse_framing():
    """Real SSE framing separates events with a blank line; iter_lines()
    yields that blank line as its own empty-string entry. A test that only
    feeds pre-filtered data: lines never exercises the skip branch."""
    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="queued"))

    lines = [
        'data: {"id": "11111111-1111-1111-1111-111111111111", "status": "running"}',
        "",
        'data: {"id": "11111111-1111-1111-1111-111111111111", "status": "completed", "result": {}}',
        "",
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.__enter__ = lambda self: mock_resp
    mock_resp.__exit__ = lambda self, *a: None

    with patch.object(c._transport._session, "get", return_value=mock_resp):
        events = list(job.stream())

    assert [e["status"] for e in events] == ["running", "completed"]


def test_sampler_job_stream_drops_mid_stream_then_falls_back_to_polling():
    """Distinct from the connect-time-failure fallback test above: here the
    SSE connection succeeds and yields a real event first, *then* drops —
    the in-progress self._snap update and the fallback must both work."""
    import requests

    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="queued"))

    def _lines_then_drop():
        yield 'data: {"id": "11111111-1111-1111-1111-111111111111", "status": "running"}'
        raise requests.exceptions.ConnectionError("dropped mid-stream")

    mock_sse_resp = MagicMock()
    mock_sse_resp.status_code = 200
    mock_sse_resp.raise_for_status = MagicMock()
    mock_sse_resp.iter_lines.return_value = _lines_then_drop()
    mock_sse_resp.__enter__ = lambda self: mock_sse_resp
    mock_sse_resp.__exit__ = lambda self, *a: None

    with patch.object(c._transport._session, "get") as mock_get:
        mock_get.side_effect = [
            mock_sse_resp,
            _mock_resp(_job_snapshot(status="completed", result={})),
        ]
        with patch("time.sleep", return_value=None):
            events = list(job.stream())

    assert [e["status"] for e in events] == ["running", "completed"]


def test_sampler_job_stream_enforces_overall_timeout_across_sse_and_polling():
    """timeout is a total wall-clock budget (matching wait()), not a
    per-request socket timeout — it must fire even after the fallback to
    polling, where the old implementation ignored it entirely."""
    import requests

    c = SnaqcsClient(api_key="snaqcs_x")
    job = SamplerJob(c, _job_snapshot(status="queued"))

    with patch.object(c._transport._session, "get", side_effect=requests.exceptions.ConnectionError("down")):
        # monotonic() called once for the deadline, then again per deadline
        # check; jump straight past the deadline on the first in-loop check.
        with patch("time.monotonic", side_effect=[0.0, 100.0]):
            with patch("time.sleep", return_value=None):
                with pytest.raises(TimeoutError):
                    list(job.stream(timeout=5.0))
