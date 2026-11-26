"""SNAQCS Python client — single entry point for all API operations."""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# One logical operator, or a list of them paired by index (k > 1). A decoder
# fails a residual that flips any listed logical; unlisted logicals are not judged.
Logicals = Union[str, List[str], None]

class AuthenticationError(Exception):
    """Raised when the API key is missing, invalid, or revoked."""


class ServerUnavailableError(Exception):
    """Raised when the SNAQCS server cannot be reached."""


class UnexpectedResponseError(Exception):
    """Raised when the server returns a non-JSON or unexpected response."""


class JobError(Exception):
    """Base class for sampler-job errors."""


class JobFailedError(JobError):
    """Raised by ``SamplerJob.wait()`` when the job ends in ``failed``."""


class JobCancelledError(JobError):
    """Raised by ``SamplerJob.wait()`` when the job ends in ``cancelled``."""


class JobNotCompletedError(JobError):
    """Raised by ``SamplerJob.result`` when the job hasn't completed yet."""


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class DecoderResult:
    """Result from decoding a single Pauli error."""
    error: str
    syndrome: tuple
    syndrome_string: str
    syndrome_weight: int
    recovery: Optional[str]
    syndrome_in_table: bool
    corrected_error: Optional[str]
    corrected_error_phase: Optional[int]
    is_correctable: bool
    correction_type: str
    final_syndrome: Optional[tuple]
    is_final_syndrome_trivial: bool
    details: str


@dataclass
class LookupTableEntry:
    """Single entry in a syndrome lookup table."""
    syndrome: tuple
    recovery: str
    syndrome_weight: int
    error_weight: int


@dataclass
class LookupTableInfo:
    """Lookup table metadata for a stabilizer code."""
    num_qubits: int
    num_stabilizers: int
    num_entries: int
    code_distance: Optional[int]
    entries: list


@dataclass
class PropagationResult:
    """Result from propagating faults through a circuit."""
    initial_error: str
    final_error: str
    has_superposition: bool
    intermediate_steps: list


# ── Transport ─────────────────────────────────────────────────────────────────

class _Transport:
    """Owns the HTTP session; the one place ConnectionError -> ServerUnavailableError
    and HTTP 401 -> AuthenticationError are translated.

    SnaqcsClient, Circuits, and Jobs/SamplerJob all send requests through the
    same _Transport instance via send() (raw Response) or send_json() (parsed
    body), instead of each re-implementing this translation.
    """

    def __init__(self, base_url: str, api_key: Optional[str], timeout: float) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self._session = requests.Session()
        retry = Retry(total=3, connect=0, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def send(
        self,
        method: str,
        path: str,
        *,
        translate_auth_errors: bool = True,
        **kwargs,
    ) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        url = f"{self.base_url}{path}"
        request = getattr(self._session, method.lower())
        try:
            resp = request(url, **kwargs)
        except requests.exceptions.ConnectionError:
            raise ServerUnavailableError(
                f"Cannot reach SNAQCS server at {self.base_url}. "
                "Check SNAQCS_API_URL or start the local dev server."
            )
        if translate_auth_errors and resp.status_code == 401:
            raise AuthenticationError(
                "Authentication failed. Pass api_key= or set SNAQCS_API_KEY env var."
            )
        return resp

    def send_json(self, method: str, path: str, **kwargs) -> Any:
        resp = self.send(method, path, **kwargs)
        resp.raise_for_status()
        if "oauth2/sign_in" in resp.url or "oauth2/auth" in resp.url:
            raise AuthenticationError(
                f"Request was redirected to OAuth2 login at {resp.url}. "
                "API key is missing or invalid — set SNAQCS_API_KEY or pass api_key=."
            )
        try:
            return resp.json()
        except requests.exceptions.JSONDecodeError:
            preview = repr(resp.text[:200])
            raise UnexpectedResponseError(
                f"Server returned HTTP {resp.status_code} with non-JSON body.\n"
                f"URL: {resp.url}\n"
                f"Content-Type: {resp.headers.get('Content-Type', 'unknown')}\n"
                f"Body (first 200 chars): {preview}"
            ) from None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_decoder_result(data: dict) -> DecoderResult:
    return DecoderResult(
        error=data["error"],
        syndrome=tuple(data["syndrome"]),
        syndrome_string=data["syndrome_string"],
        syndrome_weight=data["syndrome_weight"],
        recovery=data.get("recovery"),
        syndrome_in_table=data["syndrome_in_table"],
        corrected_error=data.get("corrected_error"),
        corrected_error_phase=data.get("corrected_error_phase"),
        is_correctable=data["is_correctable"],
        correction_type=data["correction_type"],
        final_syndrome=tuple(data["final_syndrome"]) if data.get("final_syndrome") else None,
        is_final_syndrome_trivial=data["is_final_syndrome_trivial"],
        details=data["details"],
    )


# ── Decoder helper ────────────────────────────────────────────────────────────

class Decoder:
    """
    Code-specific decoder bound to an SNAQCS client session.

    Created via ``client.decoder(...)``. Holds stabilizers and logical operators
    so you can call ``decode(error)`` repeatedly without re-specifying them.

    Example::

        decoder = client.decoder(
            stabilizers=["+XZZXI", "+IXZZX", "+XIXZZ", "+ZXIXZ"],
            num_qubits=5,
            logical_x="XXXXX",
            logical_z="ZZZZZ",
        )
        result = decoder.decode("XIIII")
        results = decoder.decode_batch(["XIIII", "IXIII", "IIXII"])

    For a code with several logical qubits, pass lists paired by index. A residual
    fails if it flips any listed logical; list a single pair to get the rate for
    that logical alone::

        both = client.decoder(stabilizers, logical_x=[lx1, lx2], logical_z=[lz1, lz2])
        first = client.decoder(stabilizers, logical_x=[lx1], logical_z=[lz1])
    """

    def __init__(
        self,
        client: "SnaqcsClient",
        stabilizers: list,
        num_qubits: Optional[int],
        logical_x: Logicals,
        logical_z: Logicals,
    ) -> None:
        self._client = client
        self.stabilizers = stabilizers
        self.num_qubits = num_qubits
        self.logical_x = logical_x
        self.logical_z = logical_z

    def decode(self, error: str) -> DecoderResult:
        """Decode a single Pauli error string."""
        return self._client.decode(
            error,
            stabilizers=self.stabilizers,
            num_qubits=self.num_qubits,
            logical_x=self.logical_x,
            logical_z=self.logical_z,
        )

    def decode_batch(self, errors: list) -> list:
        """Decode multiple Pauli error strings in a single API call."""
        return self._client.decode_batch(
            errors,
            stabilizers=self.stabilizers,
            num_qubits=self.num_qubits,
            logical_x=self.logical_x,
            logical_z=self.logical_z,
        )

    def lookup_table(self) -> LookupTableInfo:
        """Return the full syndrome → recovery lookup table for this code."""
        return self._client._lookup_table(
            stabilizers=self.stabilizers,
            num_qubits=self.num_qubits,
            logical_x=self.logical_x,
            logical_z=self.logical_z,
        )

    def __repr__(self) -> str:
        return f"Decoder(n={self.num_qubits}, stabilizers={len(self.stabilizers)})"


# ── Main client ───────────────────────────────────────────────────────────────

class SnaqcsClient:
    """
    Unified SNAQCS API client.

    Authenticate once; all methods share the same session.

    Local dev (``make start-backend`` sets SNAQCS_DISABLE_AUTH=1, no key needed)::

        client = SnaqcsClient()

    Production::

        client = SnaqcsClient(api_key="snaqcs_...")  # or set SNAQCS_API_KEY env var
        # base URL defaults to SNAQCS_API_URL env var

    Quick decode::

        result = client.decode("XIIII", stabilizers=["+XZZXI", ...], num_qubits=5)

    Code-specific decoder (Stim-like interface, reuses connection)::

        decoder = client.decoder(stabilizers=["+XZZXI", ...], num_qubits=5)
        result = decoder.decode("XIIII")
        results = decoder.decode_batch(["XIIII", "IXIII"])

    Fault propagation::

        result = client.propagate(circuit={...}, faults=[...])
        print(result.final_error, result.has_superposition)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (
            (base_url or os.environ.get("SNAQCS_API_URL") or "http://localhost:6090")
            .rstrip("/")
        )
        key = api_key or os.environ.get("SNAQCS_API_KEY")
        is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        if not key and not is_local:
            raise AuthenticationError(
                f"No API key provided for {self.base_url}. "
                "Set SNAQCS_API_KEY environment variable or pass api_key=."
            )
        self.timeout = timeout
        self._transport = _Transport(self.base_url, key, timeout)

    def _post(self, path: str, body: dict) -> Any:
        return self._transport.send_json("POST", path, json=body)

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._transport.send_json("GET", path, params=params, timeout=30)

    # ── Decoder ───────────────────────────────────────────────────────────────

    def decoder(
        self,
        stabilizers: list,
        num_qubits: Optional[int] = None,
        logical_x: Logicals = None,
        logical_z: Logicals = None,
    ) -> Decoder:
        """Return a code-specific Decoder that reuses this client's session.

        ``logical_x``/``logical_z`` take one operator or a list paired by index (k > 1).
        """
        return Decoder(self, stabilizers, num_qubits, logical_x, logical_z)

    def decode(
        self,
        error: str,
        stabilizers: list,
        num_qubits: Optional[int] = None,
        logical_x: Logicals = None,
        logical_z: Logicals = None,
    ) -> DecoderResult:
        """Decode a single Pauli error."""
        data = self._post("/api/v1/decoder/decode", {
            "error": error,
            "stabilizers": stabilizers,
            "num_qubits": num_qubits,
            "logical_x": logical_x,
            "logical_z": logical_z,
        })
        return _parse_decoder_result(data)

    def decode_batch(
        self,
        errors: list,
        stabilizers: list,
        num_qubits: Optional[int] = None,
        logical_x: Logicals = None,
        logical_z: Logicals = None,
    ) -> list:
        """Decode multiple Pauli errors in a single API call."""
        data = self._post("/api/v1/decoder/decode_batch", {
            "errors": errors,
            "stabilizers": stabilizers,
            "num_qubits": num_qubits,
            "logical_x": logical_x,
            "logical_z": logical_z,
        })
        return [_parse_decoder_result(r) for r in data["results"]]

    def _lookup_table(
        self,
        stabilizers: list,
        num_qubits: Optional[int] = None,
        logical_x: Logicals = None,
        logical_z: Logicals = None,
    ) -> LookupTableInfo:
        data = self._post("/api/v1/decoder/info", {
            "stabilizers": stabilizers,
            "num_qubits": num_qubits,
            "logical_x": logical_x,
            "logical_z": logical_z,
        })
        return LookupTableInfo(
            num_qubits=data["num_qubits"],
            num_stabilizers=data["num_stabilizers"],
            num_entries=data["num_entries"],
            code_distance=data.get("code_distance"),
            entries=[
                LookupTableEntry(
                    syndrome=tuple(e["syndrome"]),
                    recovery=e["recovery"],
                    syndrome_weight=e["syndrome_weight"],
                    error_weight=e["error_weight"],
                )
                for e in data["entries"]
            ],
        )

    # ── Faults ────────────────────────────────────────────────────────────────

    def propagate(
        self,
        circuit: dict,
        faults: list,
    ) -> PropagationResult:
        """
        Propagate faults through a circuit.

        ``faults`` is a list of ``{"location": N, "qubit": Q, "pauli": "X"}`` dicts.
        ``location=N`` means the error appears after gate N; ``location=-1`` is a prep fault.
        """
        data = self._post("/api/v1/analysis/circuit/propagate", {
            "circuit": circuit,
            "faults": faults,
        })
        return PropagationResult(
            initial_error=data["initial_error"],
            final_error=data["final_error"],
            has_superposition=data["has_superposition"],
            intermediate_steps=data.get("intermediate_steps", []),
        )

    def enumerate_faults(
        self,
        circuit: dict,
        max_fault_weight: int,
        check_functions: Dict[str, str],
        fault_types: Optional[list] = None,
        return_details: bool = False,
        sample_size: Optional[int] = None,
    ) -> dict:
        """Enumerate all fault configurations up to a given weight.

        ``check_functions`` is a dict of named simpleeval expressions evaluated
        per-fault. It goes out as ``classic_functions``, the API's name for them
        (the server still reads ``check_functions``).
        """
        body: dict = {
            "circuit": circuit,
            "max_fault_weight": max_fault_weight,
            "classic_functions": check_functions,
            "return_details": return_details,
        }
        if fault_types is not None:
            body["fault_types"] = fault_types
        if sample_size is not None:
            body["sample_size"] = sample_size
        return self._post("/api/v1/analysis/circuit/enumerate_faults", body)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        circuit: dict,
        check_functions: Dict[str, str],
        noise_config: Optional[dict] = None,
        num_samples: int = 1000,
        seed: Optional[int] = None,
        return_sample_details: bool = False,
        decoder_backend: Optional[str] = None,
        decoder_config: Optional[dict] = None,
        backend: Optional[str] = None,
    ) -> dict:
        """Monte Carlo fault sampling for a single circuit with depolarizing noise.

        ``backend="clifft"`` samples a non-Clifford circuit (T, CCX) exactly; the
        check functions then read measurement flips only (no ``final_error``, no
        ``weight``, no decoder). Omitted, the server uses the Pauli frame.
        """
        body: dict = {
            "circuit": circuit,
            "classic_functions": check_functions,
            "noise_config": noise_config or {},
            "num_samples": num_samples,
            "seed": seed,
            "return_sample_details": return_sample_details,
        }
        if decoder_backend is not None:
            body["decoder_backend"] = decoder_backend
        if decoder_config is not None:
            body["decoder_config"] = decoder_config
        if backend is not None:
            body["backend"] = backend
        return self._post("/api/v1/sampling/circuit/direct", body)

    def decode_detectors(
        self,
        circuit: dict,
        check_functions: Dict[str, str],
        *,
        noise_config: Optional[dict] = None,
        noise_ref: Optional[dict] = None,
        num_shots: int = 100_000,
        seed: Optional[int] = None,
        rounds: Optional[int] = None,
    ) -> dict:
        """Sample a detector-annotated circuit and decode it with PyMatching.

        Give exactly one noise form: ``noise_config`` inline, or ``noise_ref``
        (a library reference, sent as ``refs: {"noise": noise_ref}``). Passing
        both raises ``ValueError``.

        This call is synchronous and the server refuses runs above
        ``num_shots * num_detectors > 50_000_000`` with a 400. Larger runs go
        through ``jobs.submit_circuit_detector_decoding``.
        """
        if noise_config is not None and noise_ref is not None:
            raise ValueError("pass either noise_config or noise_ref, not both")
        body: dict = {
            "circuit": circuit,
            "classic_functions": check_functions,
            "num_shots": num_shots,
        }
        if noise_config is not None:
            body["noise_config"] = noise_config
        if noise_ref is not None:
            body["refs"] = {"noise": noise_ref}
        if seed is not None:
            body["seed"] = seed
        if rounds is not None:
            body["rounds"] = rounds
        return self._post("/api/v1/sampling/circuit/detector_decoding", body)

    def sample_protocol(
        self,
        config: dict,
        noise_config: Optional[dict] = None,
        num_samples: int = 1000,
        seed: Optional[int] = None,
        backend: Optional[str] = None,
        workers: Optional[int] = None,
        capture_failures: bool = False,
        capture_cap: Optional[int] = None,
    ) -> dict:
        """Monte Carlo fault sampling for a multi-circuit protocol.

        Runs as a ``protocol_direct_sampler`` job and blocks until it finishes;
        the server has no synchronous protocol route. Returns the job's result.
        Raises ``JobFailedError`` with the server's message when the run fails
        (an invalid protocol, an engine refusal). For long runs, use
        ``client.jobs.submit_protocol_direct_sampler`` and keep the job handle.

        ``capture_failures`` records the faults and per-visit outcomes of failing
        shots as ``failure_records``, each replayable via :meth:`replay`. Off by
        default because it costs work on every shot. ``capture_cap`` bounds how
        many are kept (lowest-weight first); the rest come back counted as
        ``failure_records_dropped``.
        """
        body = {
            "config": config,
            "noise_config": noise_config or {},
            "num_samples": num_samples,
            "seed": seed,
        }
        if backend is not None:
            body["backend"] = backend
        if workers is not None:
            body["workers"] = workers
        if capture_failures:
            body["capture_failures"] = True
        if capture_cap is not None:
            body["capture_cap"] = capture_cap
        return self.jobs.submit_protocol_direct_sampler(body).wait(poll=0.5).result

    def sample_protocol_subset(
        self,
        *,
        config: dict,
        noise_config: dict,
        susa_config: dict,
        num_samples: int = 100_000,
        seed: int | None = None,
        capture_failures: bool = False,
        capture_cap: Optional[int] = None,
    ) -> dict:
        """SUSA weight-stratified DSS sampling.

        Runs as a ``protocol_subset_sampler`` job and blocks until it finishes,
        like :meth:`sample_protocol`; for long runs keep the job handle from
        ``client.jobs.submit_protocol_subset_sampler`` instead.

        susa_config keys: max_weight (int), L_max (int), eta_max (float),
                          n_max (int, optional), backend (str, optional).
                          shots_per_task is deprecated and ignored by the server.
        Returns a superset of the direct_sampler response with extra DSS fields:
        p_logical, p_lower, p_upper, sigma_L, sigma_U, delta, eta.

        ``capture_failures`` returns ``failure_records`` for replay, each carrying
        its ``p_lower_contribution`` — this failure's share of the DSS lower bound,
        which is what a failing-shot list wants to be ranked by. Retention here is
        first-N-in-shot-order inside the engine, not lowest-weight-N.
        """
        body = {
            "config": config,
            "noise_config": noise_config,
            "num_samples": num_samples,
            "seed": seed,
            "susa_config": susa_config,
        }
        if capture_failures:
            body["capture_failures"] = True
        if capture_cap is not None:
            body["capture_cap"] = capture_cap
        return self.jobs.submit_protocol_subset_sampler(body).wait(poll=0.5).result

    # ── Fault analysis ────────────────────────────────────────────────────────

    def syndrome(
        self,
        error: dict,
        code_config: dict,
    ) -> dict:
        """
        Compute syndrome and suggested correction for an error on a CSS code.

        ``error`` is ``{"x_errors": [qubit_indices], "z_errors": [qubit_indices]}``.
        ``code_config`` is a CSSCodeConfig dict with ``x_stabilizers``, ``z_stabilizers`` etc.
        """
        return self._post("/api/v1/decoder/syndrome", {"error": error, "code": code_config})

    def propagate_multiple(
        self,
        circuit: dict,
        faults: list,
        check_functions: Dict[str, str],
        return_intermediate: bool = False,
    ) -> dict:
        """Propagate multiple simultaneous faults and evaluate check_functions."""
        return self._post("/api/v1/analysis/circuit/propagate_multi", {
            "circuit": circuit,
            "faults": faults,
            "classic_functions": check_functions,
            "return_intermediate": return_intermediate,
        })

    def wilson_ci(
        self,
        num_success: int,
        num_total: int,
        ci_z: float = 1.96,
    ) -> dict:
        """
        Wilson score confidence interval for an error rate.

        ``ci_z``: 1.0 = 1σ/68%, 1.96 = 95%, 2.0 = 2σ, 3.0 = 3σ, 5.0 = 5σ.
        Returns ``{"ci_lower": float, "ci_upper": float, "ci_z": float, "proportion": float}``.
        """
        return self._post("/api/v1/analysis/wilson_ci", {
            "num_success": num_success,
            "num_total": num_total,
            "ci_z": ci_z,
        })

    # ── Protocol ──────────────────────────────────────────────────────────────

    def propagate_protocol(
        self,
        config: dict,
        faults: list,
    ) -> dict:
        """Propagate a specific fault scenario through a multi-circuit protocol."""
        return self._post("/api/v1/analysis/protocol/propagate", {
            "config": config,
            "faults": faults,
        })

    def replayability(
        self,
        config: dict,
        check_functions: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Can this protocol's captured shots be replayed, and which readouts are gauge-dependent.

        Returns ``{"replayable": bool, "reason": str|None,
        "gauge_dependent_nodes": [...], "paths_probed": int}``.

        ``replayable: False`` is an answer, not an error: an edge predicate reads a
        raw projective outcome, so no fault list reproduces the branch it took.
        ``paths_probed`` is load-bearing — a fault set only probes the branches it
        reaches, so the claim is "no divergence observed on the paths probed",
        never a proof. A replayable protocol can still have gauge-dependent nodes,
        whose readouts are one representative of a logical class rather than the
        observed value.
        """
        body: dict = {"config": config}
        if check_functions is not None:
            body["classic_functions"] = check_functions
        return self._post("/api/v1/analysis/protocol/replayability", body)

    def replay(
        self,
        config: dict,
        record: dict,
        ignore_config_drift: bool = False,
    ) -> dict:
        """Replay one captured failing shot and return its per-visit trace.

        ``record`` is an element of a sampler's ``failure_records``. The record's
        faults drive the walk and its outcomes are pinned, so the walk follows the
        shot rather than re-deriving a branch; the verdict is echoed from the
        record, never recomputed. Returns ``trace``, ``verdict_recorded``,
        ``verdict_source``, ``record_reached_fail``, ``terminal_state_matches``,
        ``outcome_mismatch_visits`` and ``config_drift``.

        Predicates are compiled from ``config["classic_functions"]`` as strings — a
        record captured in-process with live callables cannot be replayed here.
        A record captured against a different resolved config is a 409; pass
        ``ignore_config_drift=True`` only when the difference is cosmetic.
        ``outcome_mismatch_visits`` is expected to be non-empty for a
        tableau-sourced record: that is the caveat, not a failure.
        """
        body = {"config": config, "record": record}
        if ignore_config_drift:
            body["ignore_config_drift"] = True
        return self._post("/api/v1/analysis/protocol/replay", body)

    # ── Noise ─────────────────────────────────────────────────────────────────

    def validate_noise_profile(self, profile: dict) -> dict:
        """
        Validate a hardware noise profile and convert to internal configs.

        Returns ``{"depolarizing_config": {...}, "correlated_config": ...}``.
        Raises HTTP 422 if the profile is malformed.
        """
        return self._post("/api/v1/noise_models/profile/validate", profile)

    # ── Circuit library ───────────────────────────────────────────────────────

    @property
    def circuits(self) -> "Circuits":
        """Sub-client for circuit library CRUD (``client.circuits.get(name)``, etc.)."""
        if not hasattr(self, "_circuits"):
            self._circuits = Circuits(self)
        return self._circuits

    # ── Sampler jobs ──────────────────────────────────────────────────────────

    @property
    def jobs(self) -> "Jobs":
        """Sub-client for background sampler jobs (``client.jobs.submit_protocol_direct_sampler(...)``, etc.)."""
        if not hasattr(self, "_jobs"):
            self._jobs = Jobs(self)
        return self._jobs

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Check backend connectivity. Returns server status dict.

        Public endpoint — doesn't require auth, so 401 translation is skipped.
        """
        resp = self._transport.send(
            "GET", "/api/v1/health", timeout=10, translate_auth_errors=False
        )
        resp.raise_for_status()
        return resp.json()

    def __repr__(self) -> str:
        return f"SnaqcsClient(base_url='{self.base_url}')"


# ── Circuit library sub-client ────────────────────────────────────────────────

class Circuits:
    """
    Circuit library sub-client. Access via ``client.circuits``.

    Example::

        client.circuits.save({"name": "my_encoder", "qubits": 7, "layers": [...]})
        circuit = client.circuits.get("my_encoder")
        for entry in client.circuits.list():
            print(entry["name"])
        client.circuits.delete("my_encoder")
        qasm = client.circuits.export_qasm("my_encoder")
        entry = client.circuits.import_qasm(qasm_string, name="imported")
    """

    def __init__(self, client: "SnaqcsClient") -> None:
        self._client = client
        self._transport = client._transport

    def list(
        self,
        tags: Optional[str] = None,
        search: Optional[str] = None,
    ) -> list:
        """List all circuits. Optionally filter by comma-separated ``tags`` or ``search`` string."""
        params = {}
        if tags:
            params["tags"] = tags
        if search:
            params["search"] = search
        return self._transport.send_json("GET", "/api/v1/circuits", params=params or None, timeout=30)

    def get(self, name: str) -> Optional[dict]:
        """Get a circuit by name. Returns ``None`` if not found."""
        resp = self._transport.send("GET", f"/api/v1/circuits/{name}", timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def save(self, circuit: dict) -> dict:
        """Save a new circuit to the library. ``circuit`` must include ``"name"``."""
        return self._client._post("/api/v1/circuits", circuit)

    def delete(self, name: str) -> None:
        """Delete a circuit from the library by name."""
        resp = self._transport.send("DELETE", f"/api/v1/circuits/{name}", timeout=30)
        resp.raise_for_status()

    def export_qasm(self, name: str) -> str:
        """Export a circuit as an OpenQASM string."""
        resp = self._transport.send("GET", f"/api/v1/circuits/{name}/export/qasm", timeout=30)
        resp.raise_for_status()
        return resp.text

    def import_qasm(self, qasm: str, name: Optional[str] = None) -> dict:
        """Import a circuit from an OpenQASM string. Returns the new library entry."""
        body: dict = {"qasm": qasm}
        if name:
            body["name"] = name
        return self._client._post("/api/v1/circuits/import/qasm", body)


# ── Sampler jobs sub-client ──────────────────────────────────────────────────

# Must match _TERMINAL_STATUSES in backend/app/routes/jobs.py — no shared
# source of truth across the wire, so keep these two in sync by hand.
_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}

# A deploy restarts the API container while the job keeps running in the
# worker; the edge proxy answers 502/503/504 until the API is back. The
# transport's urllib3 retries span well under a second, far shorter than a
# restart, so long-running polls ride out the outage themselves.
_TRANSIENT_HTTP_STATUSES = {502, 503, 504}
_MAX_OUTAGE_SECONDS = 600.0


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        return exc.response is not None and exc.response.status_code in _TRANSIENT_HTTP_STATUSES
    # RetryError: urllib3 exhausted its retries on a 5xx in status_forcelist.
    # ServerUnavailableError / ConnectionError: nothing listening (yet).
    return isinstance(
        exc,
        (
            ServerUnavailableError,
            requests.exceptions.RetryError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    )


class SamplerJob:
    """
    Handle for an async sampler job.

    Created by ``client.jobs.submit_protocol_direct_sampler(...)`` or
    ``client.jobs.get(id)``. Holds a snapshot plus a reference to the
    client, the same shape as ``Decoder`` — submit returns immediately,
    ``.wait()`` blocks until terminal, ``.stream()`` yields progress,
    ``.cancel()`` is best-effort.

    Example::

        job = client.jobs.submit_protocol_direct_sampler({
            "circuit": {...},
            "classic_functions": {"anyError": "weight > 0"},
            "num_samples": 10_000,
        })
        job = job.wait()
        print(job.result["decoder_summary"])
    """

    def __init__(self, client: "SnaqcsClient", snapshot: dict) -> None:
        self._client = client
        self._transport = client._transport
        self._snap = snapshot

    @property
    def id(self) -> str:
        return self._snap["id"]

    @property
    def status(self) -> str:
        return self._snap["status"]

    @property
    def progress(self) -> Optional[dict]:
        return self._snap.get("progress")

    @property
    def result(self) -> dict:
        if self._snap["status"] != "completed":
            raise JobNotCompletedError(
                f"Job {self.id} is {self._snap['status']!r}, not completed"
            )
        return self._snap["result"]

    @property
    def error(self) -> Optional[dict]:
        return self._snap.get("error")

    @property
    def submitted_at(self) -> Optional[datetime]:
        return self._parse_ts("submitted_at")

    @property
    def started_at(self) -> Optional[datetime]:
        return self._parse_ts("started_at")

    @property
    def finished_at(self) -> Optional[datetime]:
        return self._parse_ts("finished_at")

    @property
    def save_checkpoint(self) -> bool:
        return bool(self._snap.get("save_checkpoint", False))

    @property
    def resume_from_job_id(self) -> Optional[str]:
        return self._snap.get("resume_from_job_id")

    @property
    def has_checkpoint(self) -> bool:
        return bool(self._snap.get("has_checkpoint", False))

    @property
    def checkpoint_bytes(self) -> Optional[int]:
        return self._snap.get("checkpoint_bytes")

    def _parse_ts(self, field: str) -> Optional[datetime]:
        raw = self._snap.get(field)
        return datetime.fromisoformat(raw) if raw else None

    def refresh(self) -> "SamplerJob":
        """Re-fetch the current snapshot from the server."""
        self._snap = self._client._get(f"/api/v1/sampling/jobs/{self.id}")
        return self

    def _refresh_through_outage(self, deadline: Optional[float]) -> "SamplerJob":
        """``refresh()``, retrying with backoff while the server is restarting.

        Gives up with the original error after ``_MAX_OUTAGE_SECONDS`` of
        continuous outage, or with ``TimeoutError`` once ``deadline`` (a
        ``time.monotonic()`` value) passes. Non-transient errors (401, 404,
        ...) propagate immediately.
        """
        outage_start: Optional[float] = None
        delay = 1.0
        while True:
            try:
                return self.refresh()
            except Exception as exc:
                if not _is_transient(exc):
                    raise
                now = time.monotonic()
                if outage_start is None:
                    outage_start = now
                if now - outage_start >= _MAX_OUTAGE_SECONDS:
                    raise
                if deadline is not None and now >= deadline:
                    raise TimeoutError(
                        f"Job {self.id} did not finish in time; server unreachable"
                    ) from exc
                sleep_for = delay if deadline is None else min(delay, deadline - now)
                time.sleep(sleep_for)
                delay = min(delay * 2, 30.0)

    def wait(self, timeout: Optional[float] = None, poll: float = 2.0) -> "SamplerJob":
        """Block until the job reaches a terminal state, then return self.

        Raises ``JobFailedError``/``JobCancelledError`` on those terminal
        states; ``TimeoutError`` if ``timeout`` elapses first. Rides out a
        server restart (502/503/504, connection refused) of up to ten minutes.
        """
        start = time.monotonic()
        deadline = None if timeout is None else start + timeout
        while self.status not in _TERMINAL_JOB_STATUSES:
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(f"Job {self.id} did not finish within {timeout}s")
            time.sleep(poll)
            self._refresh_through_outage(deadline)
        if self.status == "failed":
            err = self.error or {}
            raise JobFailedError(err.get("message", "unknown error"))
        if self.status == "cancelled":
            raise JobCancelledError(f"Job {self.id} was cancelled")
        return self

    def stream(self, timeout: Optional[float] = None) -> Iterator[dict]:
        """Yield progress snapshots via SSE, updating ``self`` as they arrive."""
        deadline = None if timeout is None else time.monotonic() + timeout

        def _check_deadline():
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"Job {self.id} did not finish within {timeout}s")

        path = f"/api/v1/sampling/jobs/{self.id}/stream"
        try:
            with self._transport.send("GET", path, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    _check_deadline()
                    if not line or not line.startswith("data:"):
                        continue
                    snap = json.loads(line[len("data:"):].strip())
                    if "error" in snap and "status" not in snap:
                        raise JobError(f"Job {self.id} stream error: {snap['error']}")
                    self._snap = snap
                    yield snap
                    if snap.get("status") in _TERMINAL_JOB_STATUSES:
                        return
        # ServerUnavailableError: the initial connect dropped (translated by
        # _Transport.send()). requests.exceptions.ConnectionError: a drop
        # mid-stream, inside iter_lines() — after send() already returned,
        # so _Transport never gets a chance to translate it. See
        # docs/adr/0001-sampler-job-stream-error-translation.md.
        except (ServerUnavailableError, requests.exceptions.ConnectionError):
            while True:
                _check_deadline()
                self._refresh_through_outage(deadline)
                yield self._snap
                if self.status in _TERMINAL_JOB_STATUSES:
                    return
                time.sleep(2.0)

    def cancel(self, reason: Optional[str] = None) -> "SamplerJob":
        """Request cancellation. Best-effort — the worker checks at the next
        progress tick, so the job may not be ``cancelled`` immediately."""
        body = {"reason": reason} if reason else {}
        self._client._post(f"/api/v1/sampling/jobs/{self.id}/cancel", body)
        return self.refresh()

    def delete(self) -> None:
        """Soft-delete the job."""
        resp = self._transport.send("DELETE", f"/api/v1/sampling/jobs/{self.id}", timeout=30)
        resp.raise_for_status()

    def resume(self, *, save_checkpoint: Optional[bool] = None, **overrides) -> "SamplerJob":
        """Submit a new job warm-started from this one's checkpoint.

        Copies this job's original request, applies ``overrides`` (top-level
        keys like ``num_samples``; a ``susa_config`` override is merged one
        level into the original's). ``num_samples`` is cumulative — the new
        value is the total budget, not additional shots.

        ``save_checkpoint`` defaults to THIS job's own value (keeps a chain
        resumable without repeating the flag); pass it explicitly to override.
        """
        import copy
        request = copy.deepcopy(self._snap["request"])
        for key, val in overrides.items():
            if key == "susa_config" and isinstance(val, dict) and isinstance(request.get("susa_config"), dict):
                request["susa_config"] = {**request["susa_config"], **val}
            else:
                request[key] = val
        effective_save = self.save_checkpoint if save_checkpoint is None else save_checkpoint
        return self._client.jobs.submit_protocol_subset_sampler(
            request, save_checkpoint=effective_save, resume_from_job_id=self.id
        )

    def download_checkpoint(self) -> dict:
        """Download + unpack this job's saved checkpoint.

        Returns the checkpoint dict.
        Raises if the job has no saved checkpoint (server returns 404).
        """
        import gzip
        resp = self._transport.send("GET", f"/api/v1/sampling/jobs/{self.id}/checkpoint", timeout=60)
        resp.raise_for_status()
        return json.loads(gzip.decompress(resp.content))

    def __repr__(self) -> str:
        return f"SamplerJob(id={self.id!r}, status={self.status!r})"


class Jobs:
    """
    Background sampler jobs sub-client. Access via ``client.jobs``.

    Example::

        job = client.jobs.submit_protocol_direct_sampler({...})
        job.wait()
        for j in client.jobs.list(status="running"):
            print(j.id, j.progress)
    """

    def __init__(self, client: "SnaqcsClient") -> None:
        self._client = client

    def submit_protocol_direct_sampler(self, request: dict) -> SamplerJob:
        """Submit a direct-sampler job over a protocol graph.

        ``request`` is the body ``sample_protocol()`` builds: ``{"config": ...,
        "noise_config": ..., "num_samples": ..., "seed": ..., "backend": ...}``.
        """
        data = self._client._post(
            "/api/v1/sampling/jobs", {"kind": "protocol_direct_sampler", "request": request}
        )
        return self.get(data["job_id"])

    def submit_circuit_detector_decoding(self, request: dict) -> SamplerJob:
        """Submit a detector-decoding job (kind ``circuit_detector_decoding``).

        ``request`` is the body ``decode_detectors()`` posts to
        ``/api/v1/sampling/circuit/detector_decoding``: ``circuit``,
        ``classic_functions``, ``noise_config`` or ``refs.noise``, ``num_shots``,
        ``seed``, ``rounds``. Use it for runs the synchronous call refuses.
        """
        data = self._client._post(
            "/api/v1/sampling/jobs",
            {"kind": "circuit_detector_decoding", "request": request},
        )
        return self.get(data["job_id"])

    def submit_circuit_direct_sampler(self, request: dict) -> SamplerJob:
        """Submit a single-circuit direct-sampler job.

        ``request`` is the same shape ``sample()`` posts to
        ``/api/v1/sampling/circuit/direct`` — ``circuit``, ``classic_functions``,
        ``noise_config``, ``num_samples``, ``seed``, ``propagation_backend``,
        ``decoder_backend``/``decoder_config``, etc. (see
        ``SnaqcsClient.sample``'s docstring). Progress streams the same way
        as any other job — ``job.wait()`` or ``job.stream()``.
        """
        data = self._client._post(
            "/api/v1/sampling/jobs",
            {"kind": "circuit_direct_sampler", "request": request},
        )
        return self.get(data["job_id"])

    def submit_protocol_subset_sampler(
        self,
        request: dict,
        *,
        save_checkpoint: bool = False,
        resume_from_job_id: Optional[str] = None,
    ) -> SamplerJob:
        """Submit a SUSA (DSS) weight-stratified sampling job over a protocol graph.

        ``request`` is the body ``sample_protocol_subset()`` builds: ``{"config":
        ..., "noise_config": ..., "num_samples": ..., "seed": ..., "susa_config":
        {...}}``.
        ``susa_config`` requires at minimum ``max_weight`` and ``L_max``
        (``shots_per_task`` is deprecated and ignored by the server).

        ``save_checkpoint`` stores the final EventTree for later resume/inspection.
        ``resume_from_job_id`` warm-starts from a prior completed+checkpointed job.
        """
        body: dict = {"kind": "protocol_subset_sampler", "request": request,
                      "save_checkpoint": save_checkpoint}
        if resume_from_job_id is not None:
            body["resume_from_job_id"] = resume_from_job_id
        data = self._client._post("/api/v1/sampling/jobs", body)
        return self.get(data["job_id"])

    def get(self, job_id: str) -> SamplerJob:
        """Fetch a job by id."""
        return SamplerJob(self._client, self._client._get(f"/api/v1/sampling/jobs/{job_id}"))

    def list(
        self,
        status: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        """List the caller's jobs, newest first."""
        params: dict = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if kind:
            params["kind"] = kind
        data = self._client._get("/api/v1/sampling/jobs", params=params)
        return [SamplerJob(self._client, item) for item in data["items"]]


# ── Offline DSS reconstruction ───────────────────────────────────────────────
#
# Pure-Python helpers over a subset-sampler (DSS) result dict. They mirror the
# server's dss_subset_table and dss_variance modules and its reference
# dss_renewal.renewal_bracket; dss_attempt_rates is client-only. 
# tests/test_dss_reconstruct.py holds
# them to a server result at rel 1e-9.
# Heussen et al. 2024, Eqs. (2)/(4):
# p_L(p) = sum_w A_w(p) p_w, A_w = prod_i C(N_i, w_i) p_i^w_i (1 - p_i)^(N_i - w_i).


def _dss_p_vec(result: dict, p) -> list:
    """A per-partition rate vector: a list as given, a scalar scales the run's ratios."""
    if isinstance(p, (list, tuple)):
        return [float(x) for x in p]
    ref = result["p_ref_vec"]
    top = max(ref)
    return [r * float(p) / top for r in ref]


def _dss_counts(sites: dict, node: str, order: list) -> list:
    s = sites.get(node, {})
    return [int(s.get(label, 0)) for label in order]


def _dss_subset_prob(counts, w, p_vec) -> float:
    prob = 1.0
    for i in range(max(len(counts), len(w))):
        n_i = int(counts[i]) if i < len(counts) else 0
        w_i = int(w[i]) if i < len(w) else 0
        if w_i == 0 and n_i == 0:
            continue
        if w_i > n_i:
            return 0.0
        p_i = float(p_vec[i])
        prob *= math.comb(n_i, w_i) * p_i**w_i * (1.0 - p_i) ** (n_i - w_i)
    return prob


def _dss_key(path) -> tuple:
    return tuple((s["node"], tuple(s["weight"])) for s in path)


def _dss_index(rows) -> dict:
    index: dict = {}
    for r in rows:
        index.setdefault((_dss_key(r["path"]), r["node"]), []).append(r)
    return index


def _dss_require_table(result: dict, key: str = "subset_rates") -> None:
    if not result.get(key):
        raise ValueError(
            f"this result has no {key}: submit the run with "
            "include_subset_rates=True in its susa config"
        )


def _warn_if_exact_acceptance(result: dict, fn: str) -> None:
    """The server's acceptance can take its low-weight discard rate exactly from the
    census (``acceptance_low_weight == "exact"``, snaqcs 2026-11-17); the subset table
    holds only the sampled rows, so an offline rebuild gives the sampled D there."""
    if result.get("acceptance_low_weight") == "exact":
        warnings.warn(
            f"{fn}: this result's acceptance uses the census-exact low-weight discard "
            "rate, which the subset table does not hold; the rebuild uses the sampled "
            "discard rate, so its acceptance and conditioned p_L can differ from the "
            "server's p_curve.",
            stacklevel=3,
        )


def dss_p_lower(result: dict, p) -> float:
    """The server's lower end p_lower at any noise strength, from the subset table.

    ``p`` is a per-partition list (in ``partition_order``) or a scalar s, which
    scales the run's rate ratios (``p_vec = p_ref_vec * s / max(p_ref_vec)``).
    Returns F / (1 - D), conditioned on acceptance as the server's p_lower is.

    This is the LOWER end only. A subset that was never sampled contributes zero
    to it, so far from the run's p, or when no failure was sampled, it can sit
    orders of magnitude below p_L (it is exactly 0 in the second case). Use
    ``dss_bracket`` for a number to quote: it rebuilds the upper end from the
    same table.

    For a result with ``acceptance_low_weight == "exact"`` it warns: the server
    took that run's low-weight discard rate from the census, the table holds the
    sampled one.
    """
    _dss_require_table(result)
    _warn_if_exact_acceptance(result, "dss_p_lower")
    p_vec = _dss_p_vec(result, p)
    sites, order = result["fault_sites_by_node"], result["partition_order"]
    index = _dss_index(result["subset_rates"])

    def fold(key, node):
        f = d = 0.0
        for r in index.get((key, node), []):
            a = _dss_subset_prob(_dss_counts(sites, r["node"], order), r["weight"], p_vec)
            rf, rd = r["q_fail_local"], r["n_discard"] / r["n"]
            own = key + ((r["node"], tuple(r["weight"])),)
            for child, frac in r["branches"].items():
                cf, cd = fold(own, child)
                rf += frac * cf
                rd += frac * cd
            f += a * rf
            d += a * rd
        return f, d

    roots = [r for r in result["subset_rates"] if not r["path"]]
    if not roots:
        return 0.0
    F, D = fold((), roots[0]["node"])
    return F / (1.0 - D) if D else F


def _dss_comb_ratio(counts, weight, ratios) -> float:
    out = 1.0
    for n_i, w_i, r_i in zip(counts, weight, ratios):
        if w_i > n_i:
            return 0.0
        out *= math.comb(n_i, w_i) * float(r_i) ** w_i
    return out


def dss_leading_coefficient(result: dict) -> tuple:
    """``(k, c_k)``: c_k = lim p_L / s^k along the run's rate ratios, from the table.

    The counterpart of ``a`` in a fit p_L = a p^b with b = k. ``(None, None)``
    when no sampled subset failed.
    """
    _dss_require_table(result)
    sites, order = result["fault_sites_by_node"], result["partition_order"]
    ref = result["p_ref_vec"]
    ratios = [r / max(ref) for r in ref]
    rows = result["subset_rates"]
    by_key = {_dss_key(r["path"]) + ((r["node"], tuple(r["weight"])),): r for r in rows}
    coeffs: dict = {}
    for r in rows:
        if r["q_fail_local"] <= 0.0 or r["path_weight"] < 1:
            continue
        term = _dss_comb_ratio(_dss_counts(sites, r["node"], order), r["weight"], ratios)
        term *= r["q_fail_local"]
        prefix: tuple = ()
        for i, step in enumerate(r["path"]):
            prefix = prefix + ((step["node"], tuple(step["weight"])),)
            nxt = r["path"][i + 1]["node"] if i + 1 < len(r["path"]) else r["node"]
            term *= _dss_comb_ratio(_dss_counts(sites, step["node"], order), step["weight"], ratios)
            term *= by_key[prefix]["branches"].get(nxt, 0.0)
        coeffs[r["path_weight"]] = coeffs.get(r["path_weight"], 0.0) + term
    positive = sorted(k for k, c in coeffs.items() if c > 0.0)
    if not positive:
        return None, None
    return positive[0], coeffs[positive[0]]


def _dss_require_renewal(result: dict) -> None:
    if not result.get("renewal"):
        raise ValueError(result.get("renewal_note") or "this result has no renewal form")


def dss_attempt_rates(result: dict) -> list:
    """One attempt's subset table, derived from ``subset_rates``.

    An attempt runs from the start node until the shot ends or loops back to the
    start. Each row is one way an attempt can go: ``steps`` (node and weight tuple
    along it), ``attempt_weight`` (total faults), ``f`` (the fraction that ends in
    FAIL) and ``r`` (the fraction that loops back to repeat). Both are independent
    of p, so p_L(p) = sum A_s(p) f_s / (1 - sum A_s(p) r_s).

    Valid only for repeat-until-success protocols where every repeat starts fresh;
    the server decides that and reports it as ``result["renewal"]``, and this raises
    with its ``renewal_note`` otherwise.

    Unlike the attempt rows older servers sent, these carry no ``subset_prob`` (A at
    the run's p; compute A at any p from ``fault_sites_by_node`` instead) and no
    ``n`` (shots are in the matching ``subset_rates`` row).
    """
    _dss_require_renewal(result)
    _dss_require_table(result)
    rows = result["subset_rates"]
    start = next(r["node"] for r in rows if not r["path"])
    index = _dss_index(rows)
    out: list = []

    def walk(path, node, steps, frac):
        for r in index.get((_dss_key(path), node), []):
            step = {"node": r["node"], "weight": list(r["weight"])}
            st = steps + [step]
            out.append({
                "steps": st,
                "attempt_weight": sum(sum(s["weight"]) for s in st),
                "f": frac * r["q_fail_local"],               # attempt ends in FAIL
                "r": frac * r["branches"].get(start, 0.0),   # attempt loops back: repeat
            })
            for child, b in r["branches"].items():
                if child != start:                           # stay within one attempt
                    walk(path + [step], child, st, frac * b)

    walk([], start, [], 1.0)
    return out


def dss_renewal_p_lower(result: dict, p) -> float:
    """p_L = F / (1 - R) at any p from the attempt table (repeat-until-success only)."""
    attempts = dss_attempt_rates(result)
    p_vec = _dss_p_vec(result, p)
    sites, order = result["fault_sites_by_node"], result["partition_order"]
    F = R = 0.0
    for a in attempts:
        A = 1.0
        for s in a["steps"]:
            A *= _dss_subset_prob(_dss_counts(sites, s["node"], order), s["weight"], p_vec)
        F += A * a["f"]
        R += A * a["r"]
    return F / (1.0 - R)


# Port of the server's dss_variance fold (itself a port of snaqcs-susa stats.cpp).


def _dss_wilson_cov(counts):
    K, n = len(counts), sum(counts)
    pi = [float(c) / float(n) for c in counts]
    gamma = [[0.0] * K for _ in range(K)]
    if K < 2:
        return pi, gamma
    nn = float(n)
    den = (nn + 1.0) * (nn + 1.0)
    off = 1.0 / (4.0 * (K - 1))
    for a in range(K):
        gamma[a][a] = (nn * pi[a] * (1.0 - pi[a]) + 0.25) / den
        for b in range(a + 1, K):
            gamma[a][b] = gamma[b][a] = (-nn * pi[a] * pi[b] - off) / den
    return pi, gamma


def _dss_binomial(n, w, p):
    if w > n:
        return 0.0
    if n == 0:
        return 1.0
    if p <= 0.0:
        return 1.0 if w == 0 else 0.0
    if p >= 1.0:
        return 1.0 if w == n else 0.0
    return math.comb(n, w) * p**w * (1.0 - p) ** (n - w)


def _dss_up_to(counts, budget):
    out, cur = [], [0] * len(counts)

    def rec(i, rem):
        if i == len(counts):
            out.append(tuple(cur))
            return
        for v in range(min(rem, counts[i]) + 1):
            cur[i] = v
            rec(i + 1, rem - v)
        cur[i] = 0

    rec(0, budget)
    return out


def _dss_layer(counts, T):
    if not counts:
        return [()] if T == 0 else []
    out, cur = [], [0] * len(counts)

    def rec(i, rem):
        if i == len(counts) - 1:
            if rem <= counts[i] and rem <= 255:
                cur[i] = rem
                out.append(tuple(cur))
            cur[i] = 0
            return
        for v in range(min(rem, counts[i], 255) + 1):
            cur[i] = v
            rec(i + 1, rem - v)
        cur[i] = 0

    rec(0, T)
    return out


def _dss_unsampled_mass(counts, sampled, p_vec, budget):
    S = {tuple(int(x) for x in w) for w in sampled}
    mass = 0.0
    if budget >= 0:
        for w in _dss_up_to(counts, min(budget, sum(counts))):
            if w not in S:
                mass += _dss_subset_prob(counts, w, p_vec)
    pmf = [1.0]
    for n_i, p_i in zip(counts, p_vec):
        a = [_dss_binomial(n_i, w, p_i) for w in range(n_i + 1)]
        while len(a) > 1 and a[-1] == 0.0:
            a.pop()
        conv = [0.0] * (len(pmf) + len(a) - 1)
        for x, px in enumerate(pmf):
            for y, ay in enumerate(a):
                conv[x + y] += px * ay
        pmf = conv
    lo = max(budget + 1, 0)
    S_T: dict = {}
    for w in S:
        if sum(w) >= lo:
            S_T[sum(w)] = S_T.get(sum(w), 0.0) + _dss_subset_prob(counts, w, p_vec)
    over_255 = any(n > 255 for n in counts)
    for T in range(lo, max(len(pmf) - 1, max(S_T, default=-1)) + 1):
        pT = pmf[T] if T < len(pmf) else 0.0
        if T not in S_T:
            mass += pT
        elif S_T[T] <= 0.5 * pT or (T > 255 and over_255):
            mass += max(0.0, pT - S_T[T])
        else:
            mass += sum(
                _dss_subset_prob(counts, w, p_vec) for w in _dss_layer(counts, T) if w not in S
            )
    return mass


#: The level the upper-confidence mass is built for (snaqcs-susa's kConfidenceZ).
_DSS_Z = 3.0


def _dss_fold(result, p_vec, one_attempt):
    """The engine's fold over the subset table at ``p_vec``.

    Returns ``([p_fail, cutoff, p_disc, var_lo, var_hi, var_disc, cov_fd, conf], terms)``.
    ``cutoff`` is the unexplored mass and ``conf`` the upper-confidence mass of the
    sampled subsets; the server's delta is their sum. ``conf`` is the largest
    single centre shift in the tree (independent subsets: the shifts do not add),
    so it combines by max where everything else adds. With ``one_attempt`` a
    return to the start node is booked as a discard (the repeat probability R)
    instead of recursed: the fold over one attempt of a renewal protocol.
    """
    sites, order = result["fault_sites_by_node"], result["partition_order"]
    t_ft = int(result["t_ft"])
    start = next(r["node"] for r in result["subset_rates"] if not r["path"])
    repeat_to = start if one_attempt else None
    index = _dss_index(result["subset_rates"])
    terms = []

    def circuit(key, node, cum_above, mult):
        out = [0.0] * 8
        rows_here = index.get((key, node), [])
        counts = _dss_counts(sites, node, order)
        for r in rows_here:
            A = _dss_subset_prob(counts, r["weight"], p_vec)
            wf = subset(r, key, cum_above, mult * A * A)
            for i in range(3):
                out[i] += A * wf[i]
            for i in range(3, 7):
                out[i] += A * A * wf[i]
            out[7] = max(out[7], A * wf[7])
        out[1] += _dss_unsampled_mass(
            counts, [r["weight"] for r in rows_here], p_vec, t_ft - cum_above
        )
        return out

    def subset(r, key, cum_above, mult):
        cats = r["categories"]
        counts = [c["count"] for c in cats]
        K = len(cats)
        if r["fault_free"]:
            pi, gamma = [c / r["n"] for c in counts], [[0.0] * K for _ in range(K)]
        else:
            pi, gamma = _dss_wilson_cov(counts)
        own = key + ((r["node"], tuple(r["weight"])),)
        below = cum_above + sum(r["weight"])
        sub = {}
        for a, c in enumerate(cats):
            if c["kind"] == "succ" and c["child"] != repeat_to:
                sub[a] = circuit(own, c["child"], below, mult * (gamma[a][a] + pi[a] * pi[a]))
        out = [0.0] * 8
        DL, DU, DD = [0.0] * K, [0.0] * K, [0.0] * K
        for a, c in enumerate(cats):
            L = X = D = 0.0
            if c["kind"] == "succ":
                if c["child"] == repeat_to:
                    D = 1.0
                else:
                    L, X, D = sub[a][0], sub[a][1], sub[a][2]
                    out[7] = max(out[7], pi[a] * sub[a][7])
            elif c["kind"] == "fail":
                L = 1.0
            elif c["kind"] in ("unr", "trunc"):
                X = 1.0
            elif c["kind"] == "discard":
                D = 1.0
            DL[a], DU[a], DD[a] = L, L + X, D
            out[0] += pi[a] * L
            out[1] += pi[a] * X
            out[2] += pi[a] * D
        # Wilson's centre shift (z^2/n)(midpoint - mean) of the values DU, capped
        # where the mean would pass the largest one; none on the fault-free path.
        if K >= 2 and not r["fault_free"]:
            lo, hi, mean = min(DU), max(DU), out[0] + out[1]
            own_shift = min(_DSS_Z * _DSS_Z / r["n"] * (0.5 * (lo + hi) - mean), hi - mean)
            out[7] = max(out[7], own_shift)

        def quad(Dv, Ev):
            s = 0.0
            for a in range(K):
                for b in range(a + 1, K):
                    s += -gamma[a][b] * (Dv[a] - Dv[b]) * (Ev[a] - Ev[b])
            return s

        out[3], out[4], out[5], out[6] = quad(DL, DL), quad(DU, DU), quad(DD, DD), quad(DL, DD)
        terms.append((r, mult * out[3]))
        for a, s in sub.items():
            c = gamma[a][a] + pi[a] * pi[a]
            for i in range(3, 7):
                out[i] += c * s[i]
        return out

    return circuit((), start, 0, 1.0), terms


def _dss_conditioned(fold):
    """The fold's F and D as p_L = F / (1 - D), the server's ``_conditioned``.

    The upper-side mass T (cutoff plus confidence mass) may all fail or all be
    discarded; the upper end is the larger of the two. Returns
    ``(lower, sigma_lower, upper, sigma_upper)``; sigmas by the delta method.
    """
    F, cutoff, D, var_F, var_U, var_D, cov, conf = fold
    T = cutoff + conf
    sF, sU, sD = (math.sqrt(max(v, 0.0)) for v in (var_F, var_U, var_D))
    acc = max(1e-300, 1.0 - D)

    def sigma(num, s_num, den):
        var = s_num**2 / den**2 + num**2 * sD**2 / den**4 + 2 * num * cov / den**3
        return math.sqrt(max(var, 0.0))

    lower, s_lower = F / acc, sigma(F, sF, acc)
    up_fail = (F + T) / acc
    shrunk = acc - T
    up_discard = F / shrunk if shrunk > 0 else 1.0
    if up_fail >= up_discard:
        upper, s_upper = up_fail, sigma(F + T, sU, acc)
    else:
        upper, s_upper = up_discard, sigma(F, sU, max(shrunk, 1e-300))
    return lower, s_lower, min(1.0, upper), s_upper


def dss_bracket(result: dict, p) -> dict:
    """The whole DSS bracket at any noise strength, from the subset table.

    This is what to use for a p_L at another p: the lower end AND the upper end,
    rebuilt from the same rows, so the subsets the run never sampled are counted
    (in ``delta``) instead of silently contributing zero as in ``dss_p_lower``.
    It reproduces the server's ``p_curve`` point at that p; no extra shots. The
    exception is a postselected run with ``acceptance_low_weight == "exact"``:
    the server took its low-weight discard rate from the census, the table holds
    only the sampled one, so this rebuild uses the sampled discard rate and
    warns.

    ``p`` is a per-partition list (in ``partition_order``) or a scalar s, which
    scales the run's rate ratios. Keys:

    - ``p_lower``, ``sigma_L``: the lower end and its standard error.
    - ``p_upper = p_lower + delta``, ``sigma_U``.
    - ``p_upper_3sigma = p_upper + 3 sigma_U``: the bound to quote.
    - ``delta``: the mass of everything not sampled (unsampled subsets,
      unrealized successors, truncated walks) plus ``delta_confidence``, an
      upper-confidence mass for what was sampled (Wilson's centre shift at
      z = 3; it falls as 1/shots).
    - ``p_logical``: the bracket midpoint. ``p``, ``p_vec``.

    With a discard node everything is conditioned on acceptance, as the server's
    numbers are. Far from the run's p the bracket is honest but wide: the run
    chose its subsets for p_ref, and at a larger p the unsampled ones dominate.
    """
    _dss_require_table(result)
    _warn_if_exact_acceptance(result, "dss_bracket")
    if "t_ft" not in result:
        raise ValueError("this result has no t_ft: it predates the subset-rate bracket")
    p_vec = _dss_p_vec(result, p)
    fold, _ = _dss_fold(result, p_vec, one_attempt=False)
    lower, s_lower, upper, s_upper = _dss_conditioned(fold)
    delta = upper - lower
    total = fold[1] + fold[7]
    return {
        "p": max(p_vec),
        "p_vec": p_vec,
        "p_logical": (lower + upper) / 2,
        "p_lower": max(0.0, lower),
        "p_upper": upper,
        "sigma_L": s_lower,
        "sigma_U": s_upper,
        "delta": delta,
        "delta_confidence": delta * fold[7] / total if total > 0 else 0.0,
        "p_upper_3sigma": upper + 3.0 * s_upper,
    }


def dss_renewal_bracket(result: dict, p) -> dict:
    """The renewal bracket at any p (repeat-until-success only).

    Keys: p, p_vec, p_lower, p_upper, sigma_L, sigma_U, p_upper_3sigma, F, R, U
    and top_sigma_rows. U is one attempt's upper-side mass: unexplored plus the
    upper-confidence mass of its sampled subsets. M is read from
    ``result["renewal_max_attempts"]``: the lower end is a lower bound on the
    protocol as capped by its max_rounds.
    """
    _dss_require_renewal(result)
    _dss_require_table(result)
    p_vec = _dss_p_vec(result, p)
    fold, terms = _dss_fold(result, p_vec, one_attempt=True)
    F, R, var_F = fold[0], fold[2], fold[3]
    U = fold[1] + fold[7]
    lower, s_lower, upper, s_upper = _dss_conditioned(fold)
    factor = 1.0 - R ** int(result["renewal_max_attempts"])
    top = sorted(terms, key=lambda t: t[1], reverse=True)[:5]
    return {
        "p": max(p_vec),
        "p_vec": p_vec,
        "p_lower": max(0.0, lower) * factor,
        "p_upper": upper,
        "sigma_L": s_lower * factor,
        "sigma_U": s_upper,
        "p_upper_3sigma": upper + 3.0 * s_upper,
        "F": F,
        "R": R,
        "U": U,
        "top_sigma_rows": [
            {"path": r["path"], "node": r["node"], "weight": r["weight"], "share": v / var_F}
            for r, v in top
            if var_F > 0 and v > 0
        ],
    }
