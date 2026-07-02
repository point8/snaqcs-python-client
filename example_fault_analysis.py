"""
Single-fault analysis for the [[7,1,3]] Steane code encoder.

Enumerates every single-qubit fault location in the circuit, propagates each
one through the encoder, and decodes the resulting error against the real
Steane stabilizers to report which faults are actually uncorrectable.
Useful for identifying the most sensitive parts of the circuit without
running a full Monte Carlo.

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

STABILIZERS = [
    "+IIIXXXX", "+IXXIIXX", "+XIXIXIX",
    "+IIIZZZZ", "+IZZIIZZ", "+ZIZIZIZ",
]
LOGICAL_X = "XXXXXXX"
LOGICAL_Z = "ZZZZZZZ"


def main():
    client = SnaqcsClient()
    if "localhost" in client.base_url and not os.environ.get("SNAQCS_API_URL"):
        print(
            "Warning: SNAQCS_API_URL not set — connecting to local dev server.\n"
            "For cloud access run:  export SNAQCS_API_URL=https://snaqcs.point8.cloud\n"
        )
    print(f"Connected to {client.base_url}\n")

    result = client.enumerate_faults(
        circuit=CIRCUIT,
        max_fault_weight=1,
        check_functions={},
        fault_types=["X", "Y", "Z"],
        return_details=True,
    )

    decoder = client.decoder(
        stabilizers=STABILIZERS, num_qubits=7, logical_x=LOGICAL_X, logical_z=LOGICAL_Z
    )
    verdicts = decoder.decode_batch([f["final_error"] for f in result["faults"]])

    total = len(verdicts)
    uncorrectable = [
        (f, v) for f, v in zip(result["faults"], verdicts) if not v.is_correctable
    ]

    print("[[7,1,3]] Steane code — single-fault analysis")
    print("=" * 60)
    print(f"Total single faults : {total}")
    print(f"Correctable         : {total - len(uncorrectable)}  "
          f"({100*(total-len(uncorrectable))/total:.1f}%)")
    print(f"Uncorrectable       : {len(uncorrectable)}  ({100*len(uncorrectable)/total:.1f}%)")
    print()

    # Aggregate by gate location — each fault entry has a list under "fault"
    by_location: dict = {}
    for f, v in uncorrectable:
        for inner in f["fault"]:
            key = inner["location"]
            by_location.setdefault(key, []).append((inner, v))

    print("Uncorrectable fault locations:")
    print(f"  {'loc':>4}  {'count':>6}  {'paulis'}")
    print("  " + "-" * 44)
    for loc, entries in sorted(by_location.items()):
        pauli_str = ", ".join(f"q{p['qubit']}:{p['pauli']}" for p, _ in entries[:5])
        if len(entries) > 5:
            pauli_str += f" … (+{len(entries)-5} more)"
        print(f"  {loc:>4}  {len(entries):>6}  {pauli_str}")


if __name__ == "__main__":
    try:
        main()
    except ServerUnavailableError as e:
        print(f"\nError: {e}")
        raise SystemExit(1)
