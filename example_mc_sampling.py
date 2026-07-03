"""
Monte Carlo logical error rate estimation for the [[7,1,3]] Steane code,
run as a background sampler job.

Submits one async job per noise level via ``client.jobs``, waits for each to
finish, decodes against the real Steane stabilizers server-side, and prints
p_L with a 2σ Wilson confidence interval for each noise level.

Usage:
    export SNAQCS_API_KEY=snaqcs_...
    export SNAQCS_API_URL=https://snaqcs.point8.cloud
    python example_mc_sampling.py
"""

import os

from snaqcs import JobFailedError, ServerUnavailableError, SnaqcsClient

# ── Circuit: simple [[7,1,3]] Steane code encoder ─────────────────────────────
# One H gate prepares the logical qubit; six CNOT gates spread it to all data
# qubits. No MEASURE gates: decode() needs the propagated error itself, and
# MEASURE gates would consume the tracked Pauli frame before we get to see it.
CIRCUIT = {
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

STABILIZERS = [
    "+IIIXXXX", "+IXXIIXX", "+XIXIXIX",
    "+IIIZZZZ", "+IZZIIZZ", "+ZIZIZIZ",
]
LOGICAL_X = "XXXXXXX"
LOGICAL_Z = "ZZZZZZZ"

# ── Noise levels to sweep ─────────────────────────────────────────────────────
# single_qubit_gate_rate: depolarizing rate on H, T, etc.
# two_qubit_gate_rate:    depolarizing rate on CNOT
NOISE_LEVELS = [
    {"single_qubit_gate_rate": 0.001, "two_qubit_gate_rate": 0.003},
    {"single_qubit_gate_rate": 0.003, "two_qubit_gate_rate": 0.01},
    {"single_qubit_gate_rate": 0.01,  "two_qubit_gate_rate": 0.03},
]

NUM_SHOTS = 10_000
SEED = None  # set to fixed seed for reproducible results
JOB_TIMEOUT = 120.0  # seconds to wait for each job before giving up


def main():
    client = SnaqcsClient()
    if "localhost" in client.base_url and not os.environ.get("SNAQCS_API_URL"):
        print(
            "Warning: SNAQCS_API_URL not set — connecting to local dev server.\n"
            "For cloud access run:  export SNAQCS_API_URL=https://snaqcs.point8.cloud\n"
        )
    print(f"Connected to {client.base_url}\n")
    print(f"Circuit: [[7,1,3]] Steane code encoder ({NUM_SHOTS:,} shots per noise level)")
    print("p_L estimated by decoding each sample against the real Steane stabilizers.")
    print("Each noise level runs as a background job — submit returns immediately,")
    print("wait() polls until it completes.\n")
    print(f"{'p1 (1Q)':>10}  {'p2 (2Q)':>10}  {'shots':>8}  {'p_L':>8}  {'2σ CI':>20}")
    print("-" * 68)

    for noise in NOISE_LEVELS:
        job = client.jobs.submit_circuit_direct_sampler({
            "circuit": CIRCUIT,
            # check_functions is a required field on this endpoint, but we
            # only care about decoder_summary below, not this predicate.
            "check_functions": {"anyError": "weight > 0"},
            "noise_config": noise,
            "num_samples": NUM_SHOTS,
            "seed": SEED,
            "decoder_backend": "stim",
            "decoder_config": {
                "stabilizers": STABILIZERS,
                "num_qubits": 7,
                "logical_x": LOGICAL_X,
                "logical_z": LOGICAL_Z,
            },
        })
        try:
            job = job.wait(timeout=JOB_TIMEOUT, poll=1.0)
        except JobFailedError as e:
            print(f"Job {job.id} failed: {e}")
            continue
        except TimeoutError:
            print(f"Job {job.id} did not finish within {JOB_TIMEOUT}s — "
                  f"still running server-side, skipping for now.")
            continue

        summary = job.result["decoder_summary"]
        num_uncorrectable = summary["false"]
        ci = client.wilson_ci(num_success=num_uncorrectable, num_total=NUM_SHOTS, ci_z=2.0)
        p_L = num_uncorrectable / NUM_SHOTS
        ci_str = f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        print(
            f"{noise['single_qubit_gate_rate']:>10.3f}"
            f"  {noise['two_qubit_gate_rate']:>10.3f}"
            f"  {NUM_SHOTS:>8}"
            f"  {p_L:>8.4f}"
            f"  {ci_str:>20}"
        )

    print()
    print("p_L   = uncorrectable fraction, decoded inline for all NUM_SHOTS shots")
    print("2σ CI = Wilson 95% confidence interval")


if __name__ == "__main__":
    try:
        main()
    except ServerUnavailableError as e:
        print(f"\nError: {e}")
        raise SystemExit(1)
