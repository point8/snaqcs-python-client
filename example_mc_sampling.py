"""
Monte Carlo logical error rate estimation for the [[7,1,3]] Steane code.

Runs the direct sampler at several noise levels and prints p_L with a 2σ
Wilson confidence interval for each. This is the fastest way to get a
threshold curve from the SNAQCS API.

Usage:
    export SNAQCS_API_KEY=snaqcs_...
    export SNAQCS_API_URL=https://snaqcs.point8.cloud
    python example_mc_sampling.py
"""

import os

from snaqcs import SnaqcsClient, ServerUnavailableError

# ── Circuit: simple [[7,1,3]] Steane code encoder ─────────────────────────────
# One H gate prepares the logical qubit; six CNOT gates spread it to all data qubits.
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

# ── Code: [[7,1,3]] Steane code ───────────────────────────────────────────────
CODE = {
    "n": 7,
    "k": 1,
    "d": 3,
    "x_stabilizers": ["IIIXXXX", "IXXIIXX", "XIXIXIX"],
    "z_stabilizers": ["IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"],
    "logical_x": ["XXXXXXX"],
    "logical_z": ["ZZZZZZZ"],
}

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


def main():
    client = SnaqcsClient()
    if "localhost" in client.base_url and not os.environ.get("SNAQCS_API_URL"):
        print(
            "Warning: SNAQCS_API_URL not set — connecting to local dev server.\n"
            "For cloud access run:  export SNAQCS_API_URL=https://snaqcs.point8.cloud\n"
        )
    print(f"Connected to {client.base_url}\n")
    print(f"Circuit:    [[7,1,3]] Steane code encoder ({NUM_SHOTS:,} shots per noise level)")
    print(f"{'p1 (1Q)':>10}  {'p2 (2Q)':>10}  {'p_L':>8}  {'2σ CI':>20}  {'uncorrectable':>14}")
    print("-" * 72)

    for noise in NOISE_LEVELS:
        result = client.sample(
            circuit=CIRCUIT,
            code_config=CODE,
            noise_config=noise,
            num_samples=NUM_SHOTS,
            seed=SEED,
        )
        ci = client.wilson_ci(
            num_success=result["num_uncorrectable"],
            num_total=NUM_SHOTS,
            ci_z=2.0,
        )
        p_L = result["uncorrectable_fraction"]
        ci_str = f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        print(
            f"{noise['single_qubit_gate_rate']:>10.3f}"
            f"  {noise['two_qubit_gate_rate']:>10.3f}"
            f"  {p_L:>8.4f}"
            f"  {ci_str:>20}"
            f"  {result['num_uncorrectable']:>8} / {NUM_SHOTS:,}"
        )

    print()
    print("p_L  = logical error rate per shot")
    print("2σ CI = Wilson 95% confidence interval")


if __name__ == "__main__":
    try:
        main()
    except ServerUnavailableError as e:
        print(f"\nError: {e}")
        raise SystemExit(1)
