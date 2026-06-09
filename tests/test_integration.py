"""Integration tests against a locally running SNAQCS dev server.

Skipped automatically when the server is not reachable — safe to run alongside
the unit tests at any time.

Start the server with:
    make start-backend   # → http://localhost:6090 (auth disabled)

Then run:
    uv run pytest tests/test_integration.py -v
"""

import pytest
import requests

from snaqcs import SnaqcsClient, AuthenticationError, DecoderResult, PropagationResult

BASE_URL = "http://localhost:6090"

# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """SnaqcsClient pointed at local dev server. Skips if server is down."""
    try:
        requests.get(f"{BASE_URL}/api/health", timeout=2)
    except requests.exceptions.ConnectionError:
        pytest.skip("Local dev server not running at http://localhost:6090")
    return SnaqcsClient(base_url=BASE_URL)


@pytest.fixture(scope="session")
def steane_code():
    return {
        "n": 7,
        "k": 1,
        "d": 3,
        "x_stabilizers": ["IIIXXXX", "IXXIIXX", "XIXIXIX"],
        "z_stabilizers": ["IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"],
        "logical_x": ["XXXXXXX"],
        "logical_z": ["ZZZZZZZ"],
    }


@pytest.fixture(scope="session")
def steane_stabilizers():
    return [
        "+IIIXXXX", "+IXXIIXX", "+XIXIXIX",
        "+IIIZZZZ", "+IZZIIZZ", "+ZIZIZIZ",
    ]


@pytest.fixture(scope="session")
def steane_circuit():
    return {
        "qubits": 7,
        "layers": [
            [{"gate": "h", "qubits": [0]}],
            [{"gate": "cx", "qubits": [0, 1]}],
            [{"gate": "cx", "qubits": [0, 2]}],
            [{"gate": "cx", "qubits": [0, 3]}],
            [{"gate": "cx", "qubits": [0, 4]}],
            [{"gate": "cx", "qubits": [0, 5]}],
            [{"gate": "cx", "qubits": [0, 6]}],
        ],
    }


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_dict(self, client):
        result = client.health()
        assert isinstance(result, dict)

    def test_no_auth_required(self, client):
        # Health endpoint is public — must work without a key
        c = SnaqcsClient(base_url=BASE_URL)
        result = c.health()
        assert result is not None


# ── Decoder ───────────────────────────────────────────────────────────────────

class TestDecode:
    def test_single_qubit_x_error_correctable(self, client, steane_stabilizers):
        result = client.decode(
            "XIIIIII",
            stabilizers=steane_stabilizers,
            num_qubits=7,
            logical_x="XXXXXXX",
            logical_z="ZZZZZZZ",
        )
        assert isinstance(result, DecoderResult)
        assert result.is_correctable is True
        assert result.syndrome_weight > 0
        assert result.is_final_syndrome_trivial is True

    def test_logical_x_not_correctable(self, client, steane_stabilizers):
        result = client.decode(
            "XXXXXXX",
            stabilizers=steane_stabilizers,
            num_qubits=7,
            logical_x="XXXXXXX",
            logical_z="ZZZZZZZ",
        )
        assert result.is_correctable is False

    def test_identity_correctable(self, client, steane_stabilizers):
        result = client.decode(
            "IIIIIII",
            stabilizers=steane_stabilizers,
            num_qubits=7,
            logical_x="XXXXXXX",
            logical_z="ZZZZZZZ",
        )
        assert result.is_correctable is True
        assert result.syndrome_weight == 0

    def test_batch_all_single_qubit_x_errors_correctable(self, client, steane_stabilizers):
        errors = [("I" * q + "X" + "I" * (6 - q)) for q in range(7)]
        results = client.decode_batch(
            errors,
            stabilizers=steane_stabilizers,
            num_qubits=7,
            logical_x="XXXXXXX",
            logical_z="ZZZZZZZ",
        )
        assert len(results) == 7
        assert all(r.is_correctable for r in results)

    def test_decoder_helper_reuses_session(self, client, steane_stabilizers):
        decoder = client.decoder(
            stabilizers=steane_stabilizers,
            num_qubits=7,
            logical_x="XXXXXXX",
            logical_z="ZZZZZZZ",
        )
        result = decoder.decode("ZIIIIII")
        assert isinstance(result, DecoderResult)
        assert result.is_correctable is True


# ── MC Sampling ───────────────────────────────────────────────────────────────

class TestSample:
    def test_zero_noise_zero_failures(self, client, steane_circuit, steane_code):
        result = client.sample(
            circuit=steane_circuit,
            code_config=steane_code,
            noise_config={"single_qubit_gate_rate": 0.0, "two_qubit_gate_rate": 0.0},
            num_samples=200,
            seed=42,
        )
        assert result["num_samples"] == 200
        assert result["num_uncorrectable"] == 0

    def test_high_noise_produces_failures(self, client, steane_circuit, steane_code):
        result = client.sample(
            circuit=steane_circuit,
            code_config=steane_code,
            noise_config={"single_qubit_gate_rate": 0.2, "two_qubit_gate_rate": 0.2},
            num_samples=500,
            seed=42,
        )
        assert result["num_uncorrectable"] > 0

    def test_response_counts_are_consistent(self, client, steane_circuit, steane_code):
        result = client.sample(
            circuit=steane_circuit,
            code_config=steane_code,
            noise_config={"single_qubit_gate_rate": 0.01, "two_qubit_gate_rate": 0.01},
            num_samples=300,
            seed=0,
        )
        assert result["num_correctable"] + result["num_uncorrectable"] == result["num_samples"]
        assert 0.0 <= result["uncorrectable_fraction"] <= 1.0

    def test_seed_is_deterministic(self, client, steane_circuit, steane_code):
        kwargs = dict(
            circuit=steane_circuit,
            code_config=steane_code,
            noise_config={"single_qubit_gate_rate": 0.01, "two_qubit_gate_rate": 0.01},
            num_samples=500,
            seed=99,
        )
        r1 = client.sample(**kwargs)
        r2 = client.sample(**kwargs)
        assert r1["num_uncorrectable"] == r2["num_uncorrectable"]


# ── Fault enumeration ─────────────────────────────────────────────────────────

class TestEnumerateSingleFaults:
    def test_returns_expected_fields(self, client, steane_circuit, steane_code):
        result = client.enumerate_single_faults(
            circuit=steane_circuit,
            code_config=steane_code,
        )
        assert "total_faults" in result
        assert "faults" in result
        assert "correctable" in result
        assert "uncorrectable" in result
        assert "uncorrectable_fraction" in result

    def test_fault_count_plausible(self, client, steane_circuit, steane_code):
        result = client.enumerate_single_faults(
            circuit=steane_circuit,
            code_config=steane_code,
        )
        # 7 qubits × 7 layers × 3 Paulis = 147 max; real count depends on gate structure
        assert result["total_faults"] > 0
        assert result["correctable"] + result["uncorrectable"] == result["total_faults"]

    def test_each_fault_has_required_fields(self, client, steane_circuit, steane_code):
        result = client.enumerate_single_faults(
            circuit=steane_circuit,
            code_config=steane_code,
        )
        for fault in result["faults"]:
            assert "location" in fault
            assert "qubit" in fault
            assert "pauli" in fault
            assert "is_correctable" in fault
            assert fault["pauli"] in ("X", "Y", "Z")
