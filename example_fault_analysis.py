"""
Single-fault analysis for the [[7,1,3]] Steane code encoder.

Enumerates every single-qubit fault location in the circuit, propagates each
one through the encoder, and reports which gate positions produce correctable
vs uncorrectable logical errors. Useful for identifying the most sensitive
parts of the circuit without running a full Monte Carlo.

Usage:
    export SNAQCS_API_KEY=snaqcs_...
    export SNAQCS_API_URL=https://snaqcs.point8.cloud
    python example_fault_analysis.py
"""

import os

from snaqcs import SnaqcsClient, ServerUnavailableError

# ── Same circuit and code as example_mc_sampling.py ──────────────────────────
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

CODE = {
    "n": 7,
    "k": 1,
    "d": 3,
    "x_stabilizers": ["IIIXXXX", "IXXIIXX", "XIXIXIX"],
    "z_stabilizers": ["IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"],
    "logical_x": ["XXXXXXX"],
    "logical_z": ["ZZZZZZZ"],
}


def main():
    client = SnaqcsClient()
    if "localhost" in client.base_url and not os.environ.get("SNAQCS_API_URL"):
        print(
            "Warning: SNAQCS_API_URL not set — connecting to local dev server.\n"
            "For cloud access run:  export SNAQCS_API_URL=https://snaqcs.point8.cloud\n"
        )
    print(f"Connected to {client.base_url}\n")

    result = client.enumerate_single_faults(
        circuit=CIRCUIT,
        code_config=CODE,
        fault_types=["X", "Y", "Z"],
    )

    total = result["total_faults"]
    correctable = result["correctable"]
    uncorrectable = result["uncorrectable"]
    p_L = result["uncorrectable_fraction"]

    print("[[7,1,3]] Steane code — single-fault analysis")
    print("=" * 60)
    print(f"Total fault locations : {total}")
    print(f"Correctable           : {correctable}  ({100*correctable/total:.1f}%)")
    print(f"Uncorrectable         : {uncorrectable}  ({100*uncorrectable/total:.1f}%)")
    print(f"p_L (fault-rate model): {p_L:.4f}")
    print()

    # Group uncorrectable faults by gate location for easier inspection
    uncorrectable_faults = [f for f in result["faults"] if not f["is_correctable"]]
    if not uncorrectable_faults:
        print("All single faults are correctable.")
        return

    # Aggregate by (location, gate_name)
    by_location: dict = {}
    for f in uncorrectable_faults:
        key = (f["location"], f.get("gate_name", "?"))
        by_location.setdefault(key, []).append(f)

    print("Uncorrectable faults by gate location:")
    print(f"  {'loc':>4}  {'gate':>6}  {'count':>6}  {'paulis'}")
    print("  " + "-" * 44)
    for (loc, gate), faults in sorted(by_location.items()):
        paulis = ", ".join(
            f"q{f['qubit']}:{f['pauli']}" for f in faults[:5]
        )
        if len(faults) > 5:
            paulis += f" … (+{len(faults)-5} more)"
        print(f"  {loc:>4}  {gate:>6}  {len(faults):>6}  {paulis}")

    print()
    print("loc  = gate index in the circuit (0-based)")
    print("q    = qubit index, pauli = error type (X/Y/Z)")


if __name__ == "__main__":
    try:
        main()
    except ServerUnavailableError as e:
        print(f"\nError: {e}")
        raise SystemExit(1)
