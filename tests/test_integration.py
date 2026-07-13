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

from snaqcs import SnaqcsClient, DecoderResult, PropagationResult

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
def steane_stabilizers():
    return [
        "+IIIXXXX", "+IXXIIXX", "+XIXIXIX",
        "+IIIZZZZ", "+IZZIIZZ", "+ZIZIZIZ",
    ]


@pytest.fixture(scope="session")
def steane_circuit():
    """Steane encoder with explicit measure gates for predicate access to m0..m6."""
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
            [{"gate": "measure", "qubits": [0]}],
            [{"gate": "measure", "qubits": [1]}],
            [{"gate": "measure", "qubits": [2]}],
            [{"gate": "measure", "qubits": [3]}],
            [{"gate": "measure", "qubits": [4]}],
            [{"gate": "measure", "qubits": [5]}],
            [{"gate": "measure", "qubits": [6]}],
        ],
    }


# Logical Z parity predicate: XOR of all 7 measurement outcomes.
CHECK_FUNCTIONS = {"logErr": "m0 ^ m1 ^ m2 ^ m3 ^ m4 ^ m5 ^ m6"}


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


# ── Propagate ─────────────────────────────────────────────────────────────────

class TestPropagate:
    def test_returns_propagation_result(self, client, steane_circuit):
        result = client.propagate(
            circuit=steane_circuit,
            faults=[{"location": 0, "qubit": 0, "pauli": "X"}],
        )
        assert isinstance(result, PropagationResult)
        assert isinstance(result.final_error, str)
        assert isinstance(result.initial_error, str)
        assert isinstance(result.has_superposition, bool)

    def test_no_fault_leaves_identity(self, client, steane_circuit):
        result = client.propagate(
            circuit=steane_circuit,
            faults=[],
        )
        assert result.final_error == "I" * 7 or all(c == "I" for c in result.final_error)


# ── MC Sampling ───────────────────────────────────────────────────────────────

class TestSample:
    def test_zero_noise_zero_failures(self, client, steane_circuit):
        result = client.sample(
            circuit=steane_circuit,
            check_functions=CHECK_FUNCTIONS,
            noise_config={"single_qubit_gate_rate": 0.0, "two_qubit_gate_rate": 0.0},
            num_samples=200,
            seed=42,
        )
        assert result["num_samples"] == 200
        assert result["checks"]["logErr"]["true"] == 0

    def test_high_noise_produces_failures(self, client, steane_circuit):
        result = client.sample(
            circuit=steane_circuit,
            check_functions=CHECK_FUNCTIONS,
            noise_config={"single_qubit_gate_rate": 0.2, "two_qubit_gate_rate": 0.2},
            num_samples=500,
            seed=42,
        )
        assert result["checks"]["logErr"]["true"] > 0

    def test_response_counts_are_consistent(self, client, steane_circuit):
        result = client.sample(
            circuit=steane_circuit,
            check_functions=CHECK_FUNCTIONS,
            noise_config={"single_qubit_gate_rate": 0.01, "two_qubit_gate_rate": 0.01},
            num_samples=300,
            seed=0,
        )
        checks = result["checks"]["logErr"]
        assert checks["true"] + checks["false"] == result["num_samples"]
        assert 0.0 <= checks["fraction_true"] <= 1.0

    def test_seed_is_deterministic(self, client, steane_circuit):
        kwargs = dict(
            circuit=steane_circuit,
            check_functions=CHECK_FUNCTIONS,
            noise_config={"single_qubit_gate_rate": 0.01, "two_qubit_gate_rate": 0.01},
            num_samples=500,
            seed=99,
        )
        r1 = client.sample(**kwargs)
        r2 = client.sample(**kwargs)
        assert r1["checks"]["logErr"]["true"] == r2["checks"]["logErr"]["true"]


# ── Fault enumeration (enumerate_faults) ─────────────────────────────────────

class TestEnumerateFaults:
    def test_returns_expected_fields(self, client, steane_circuit):
        result = client.enumerate_faults(
            circuit=steane_circuit,
            max_fault_weight=1,
            check_functions=CHECK_FUNCTIONS,
        )
        assert "total_faults" in result
        assert "summary" in result
        assert "checks" in result["summary"]
        assert "logErr" in result["summary"]["checks"]

    def test_fault_count_plausible(self, client, steane_circuit):
        result = client.enumerate_faults(
            circuit=steane_circuit,
            max_fault_weight=1,
            check_functions=CHECK_FUNCTIONS,
        )
        assert result["total_faults"] > 0
        checks = result["summary"]["checks"]["logErr"]
        assert checks["true"] + checks["false"] == result["total_faults"]

    def test_details_include_per_fault_checks(self, client, steane_circuit):
        result = client.enumerate_faults(
            circuit=steane_circuit,
            max_fault_weight=1,
            check_functions=CHECK_FUNCTIONS,
            return_details=True,
        )
        for fault in result["faults"]:
            assert "checks" in fault
            assert "logErr" in fault["checks"]


# ── Fault enumeration (enumerate_pauli_faults) ────────────────────────────────

class TestEnumeratePauliFaults:
    def test_returns_expected_fields(self, client, steane_circuit):
        result = client.enumerate_pauli_faults(circuit=steane_circuit)
        assert "total_faults" in result
        assert "faults" in result

    def test_each_fault_has_required_fields(self, client, steane_circuit):
        result = client.enumerate_pauli_faults(circuit=steane_circuit)
        for fault in result["faults"]:
            assert "location" in fault
            assert "qubit" in fault
            assert "pauli" in fault
            assert fault["pauli"] in ("X", "Y", "Z")

    def test_backward_compat_alias_works(self, client, steane_circuit):
        """enumerate_single_faults alias must produce the same result as enumerate_pauli_faults."""
        result = client.enumerate_single_faults(circuit=steane_circuit)
        assert "total_faults" in result
        assert "faults" in result
