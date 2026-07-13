"""
DSS (SUSA) vs. DirectSampler 

Usage:
    uv run python example_bell_pair_dss_vs_direct.py
"""
import math

from snaqcs import JobFailedError, ServerUnavailableError, SnaqcsClient

TWO_Q_RATE = 0.05
ANALYTIC_P_FAIL = TWO_Q_RATE * 8 / 15

CIRCUIT = {
    "qubits": 2,
    "layers": [
        [{"gate": "h", "qubits": [0]}],
        [{"gate": "cx", "qubits": [0, 1]}],
        [{"gate": "measure", "qubits": [0]}, {"gate": "measure", "qubits": [1]}],
    ],
}

PROTOCOL_CONFIG = {
    "circuits": {
        "start": {"name": "start", "kind": "start"},
        "circ": {"name": "circ", "kind": "circuit", "circuit": CIRCUIT},
        "DONE": {"name": "DONE", "kind": "done"},
        "FAIL": {"name": "FAIL", "kind": "fail"},
    },
    "edges": [
        {"from_node": "start", "to_node": "circ", "check": "True"},
        {"from_node": "circ", "to_node": "FAIL", "check": "circ in ('01', '10')"},
        {"from_node": "circ", "to_node": "DONE", "check": "circ in ('00', '11')"},
    ],
}

NOISE_CONFIG = {"two_qubit_gate_rate": TWO_Q_RATE}


def wald_sigma(p_hat, n):
    return math.sqrt(p_hat * (1 - p_hat) / n) if 0 < p_hat < 1 else 0.0


def main():
    client = SnaqcsClient()
    print(f"Connected to {client.base_url}\n")

    print("Submitting SUSA (DSS) job...")
    susa_job = client.jobs.submit_subset_sampler({
        "config": PROTOCOL_CONFIG,
        "noise_config": NOISE_CONFIG,
        "num_samples": 5_000,
        "seed": 1,
        "susa_config": {
            "max_weight": 1, 
            "shots_per_task": 1,
            "L_max": 1,
            "eta_max": 2e-3,
            "k_batch": 200,
            "criterion": "eru",
        "workers": 1,    
        },
    }).wait(timeout=60.0, poll=1.0)

    print("Submitting DirectSampler job...")
    direct_job = client.jobs.submit_direct_sampler({
        "config": PROTOCOL_CONFIG,
        "noise_config": NOISE_CONFIG,
        "num_samples": 50_000,
        "seed": 1,
        "backend": "tableau",
    }).wait(timeout=60.0, poll=1.0)

    susa, direct = susa_job.result, direct_job.result

    print("\n SUSA (DSS) result:")
    for field in ("p_logical", "p_lower", "p_upper", "sigma_L", "sigma_U",
                  "delta", "eta", "num_samples"):
        print(f"  {field:12s} = {susa[field]}")

    print("\n DirectSampler result:")
    for field in ("num_samples", "num_uncorrectable", "uncorrectable_fraction"):
        print(f"  {field:22s} = {direct[field]}")

    print(f"\n Analytic p_logical (0.05 * 8/15) = {ANALYTIC_P_FAIL:.6f}")

    direct_sigma = wald_sigma(direct["uncorrectable_fraction"], direct["num_samples"])
    susa_sigma = (susa["sigma_L"] + susa["sigma_U"]) / 2

    print("\nComparison:")
    print(f"  SUSA (DSS):  p_logical = {susa['p_logical']:.4f}  "
          f"sigma = {susa_sigma:.4f}  shots = {susa['num_samples']:,}")
    print(f"  Direct MC:   p_logical = {direct['uncorrectable_fraction']:.4f}  "
          f"sigma = {direct_sigma:.4f}  shots = {direct['num_samples']:,}")
    print(f"  Analytic:    p_logical = {ANALYTIC_P_FAIL:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (ServerUnavailableError, JobFailedError) as e:
        raise SystemExit(str(e))
