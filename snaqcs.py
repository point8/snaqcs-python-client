"""SNAQCS Python client — single entry point for all API operations."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_session(api_key: Optional[str]) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, connect=0, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    if api_key:
        session.headers["Authorization"] = f"Bearer {api_key}"
    return session


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
    """

    def __init__(
        self,
        client: "SnaqcsClient",
        stabilizers: list,
        num_qubits: Optional[int],
        logical_x: Optional[str],
        logical_z: Optional[str],
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
        self._session = _make_session(key)
        self.timeout = timeout

    def _parse_json(self, resp: requests.Response) -> Any:
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

    def _post(self, path: str, body: dict) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(url, json=body, timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            raise ServerUnavailableError(
                f"Cannot reach SNAQCS server at {self.base_url}. "
                "Check SNAQCS_API_URL or start the local dev server."
            )
        if resp.status_code == 401:
            raise AuthenticationError(
                "Authentication failed. Pass api_key= or set SNAQCS_API_KEY env var."
            )
        resp.raise_for_status()
        return self._parse_json(resp)

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        try:
            resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=30)
        except requests.exceptions.ConnectionError:
            raise ServerUnavailableError(
                f"Cannot reach SNAQCS server at {self.base_url}. "
                "Check SNAQCS_API_URL or start the local dev server."
            )
        if resp.status_code == 401:
            raise AuthenticationError(
                "Authentication failed. Pass api_key= or set SNAQCS_API_KEY env var."
            )
        resp.raise_for_status()
        return self._parse_json(resp)

    # ── Decoder ───────────────────────────────────────────────────────────────

    def decoder(
        self,
        stabilizers: list,
        num_qubits: Optional[int] = None,
        logical_x: Optional[str] = None,
        logical_z: Optional[str] = None,
    ) -> Decoder:
        """Return a code-specific Decoder that reuses this client's session."""
        return Decoder(self, stabilizers, num_qubits, logical_x, logical_z)

    def decode(
        self,
        error: str,
        stabilizers: list,
        num_qubits: Optional[int] = None,
        logical_x: Optional[str] = None,
        logical_z: Optional[str] = None,
    ) -> DecoderResult:
        """Decode a single Pauli error."""
        data = self._post("/api/decode", {
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
        logical_x: Optional[str] = None,
        logical_z: Optional[str] = None,
    ) -> list:
        """Decode multiple Pauli errors in a single API call."""
        data = self._post("/api/decode/batch", {
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
        logical_x: Optional[str] = None,
        logical_z: Optional[str] = None,
    ) -> LookupTableInfo:
        data = self._post("/api/decode/info", {
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
        data = self._post("/api/propagate", {
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
        per-fault 
        """
        body: dict = {
            "circuit": circuit,
            "max_fault_weight": max_fault_weight,
            "check_functions": check_functions,
            "return_details": return_details,
        }
        if fault_types is not None:
            body["fault_types"] = fault_types
        if sample_size is not None:
            body["sample_size"] = sample_size
        return self._post("/api/enumerate_faults", body)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        circuit: dict,
        check_functions: Dict[str, str],
        noise_config: Optional[dict] = None,
        num_samples: int = 1000,
        seed: Optional[int] = None,
        return_sample_details: bool = False,
        propagation_backend: Optional[str] = None,
        decoder_backend: Optional[str] = None,
        decoder_config: Optional[dict] = None,
    ) -> dict:
        """Monte Carlo fault sampling for a single circuit with depolarizing noise.
        """
        body: dict = {
            "circuit": circuit,
            "check_functions": check_functions,
            "noise_config": noise_config or {},
            "num_samples": num_samples,
            "seed": seed,
            "return_sample_details": return_sample_details,
        }
        if propagation_backend is not None:
            body["propagation_backend"] = propagation_backend
        if decoder_backend is not None:
            body["decoder_backend"] = decoder_backend
        if decoder_config is not None:
            body["decoder_config"] = decoder_config
        return self._post("/api/direct_sampler", body)

    def sample_protocol(
        self,
        config: dict,
        noise_config: Optional[dict] = None,
        num_samples: int = 1000,
        seed: Optional[int] = None,
        backend: Optional[str] = None,
        workers: Optional[int] = None,
    ) -> dict:
        """Monte Carlo fault sampling for a multi-circuit protocol.
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
        return self._post("/api/protocol/direct_sampler", body)

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
        return self._post("/api/syndrome", {"error": error, "code": code_config})

    def propagate_multiple(
        self,
        circuit: dict,
        faults: list,
        check_functions: Dict[str, str],
        return_intermediate: bool = False,
    ) -> dict:
        """Propagate multiple simultaneous faults and evaluate check_functions."""
        return self._post("/api/multiple_faults", {
            "circuit": circuit,
            "faults": faults,
            "check_functions": check_functions,
            "return_intermediate": return_intermediate,
        })

    def enumerate_pauli_faults(
        self,
        circuit: dict,
        code_config: Optional[dict] = None,
        fault_types: Optional[list] = None,
        include_idle: bool = False,
    ) -> dict:
        """Enumerate every single-qubit Pauli error in the circuit.
        """
        return self._post("/api/enumerate_pauli_faults", {
            "circuit": circuit,
            "code_config": code_config,
            "fault_types": fault_types or ["X", "Y", "Z"],
            "include_idle": include_idle,
        })

    # Backward-compat alias (one-release deprecation; remove next major).
    enumerate_single_faults = enumerate_pauli_faults

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
        return self._post("/api/wilson_ci", {
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
        return self._post("/api/protocol/propagate", {
            "config": config,
            "faults": faults,
        })

    # ── Noise ─────────────────────────────────────────────────────────────────

    def validate_noise_profile(self, profile: dict) -> dict:
        """
        Validate a hardware noise profile and convert to internal configs.

        Returns ``{"depolarizing_config": {...}, "correlated_config": ...}``.
        Raises HTTP 422 if the profile is malformed.
        """
        return self._post("/api/noise/profile/validate", profile)

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
        """Sub-client for background sampler jobs (``client.jobs.submit_direct_sampler(...)``, etc.)."""
        if not hasattr(self, "_jobs"):
            self._jobs = Jobs(self)
        return self._jobs

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Check backend connectivity. Returns server status dict."""
        try:
            resp = self._session.get(f"{self.base_url}/api/health", timeout=10)
        except requests.exceptions.ConnectionError:
            raise ServerUnavailableError(
                f"Cannot reach SNAQCS server at {self.base_url}. "
                "Check SNAQCS_API_URL or start the local dev server."
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

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        resp = self._client._session.get(
            f"{self._client.base_url}{path}", params=params, timeout=30
        )
        if resp.status_code == 401:
            raise AuthenticationError(
                "Authentication failed. Pass api_key= or set SNAQCS_API_KEY env var."
            )
        resp.raise_for_status()
        return resp.json()

    def _get_raw(self, path: str) -> Any:
        resp = self._client._session.get(
            f"{self._client.base_url}{path}", timeout=30
        )
        if resp.status_code == 401:
            raise AuthenticationError(
                "Authentication failed. Pass api_key= or set SNAQCS_API_KEY env var."
            )
        return resp

    def _delete(self, path: str) -> None:
        resp = self._client._session.delete(
            f"{self._client.base_url}{path}", timeout=30
        )
        if resp.status_code == 401:
            raise AuthenticationError(
                "Authentication failed. Pass api_key= or set SNAQCS_API_KEY env var."
            )
        resp.raise_for_status()

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
        return self._get("/api/circuits", params=params or None)

    def get(self, name: str) -> Optional[dict]:
        """Get a circuit by name. Returns ``None`` if not found."""
        resp = self._get_raw(f"/api/circuits/{name}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def save(self, circuit: dict) -> dict:
        """Save a new circuit to the library. ``circuit`` must include ``"name"``."""
        return self._client._post("/api/circuits", circuit)

    def delete(self, name: str) -> None:
        """Delete a circuit from the library by name."""
        self._delete(f"/api/circuits/{name}")

    def export_qasm(self, name: str) -> str:
        """Export a circuit as an OpenQASM string."""
        resp = self._get_raw(f"/api/circuits/{name}/export/qasm")
        resp.raise_for_status()
        return resp.text

    def import_qasm(self, qasm: str, name: Optional[str] = None) -> dict:
        """Import a circuit from an OpenQASM string. Returns the new library entry."""
        body: dict = {"qasm": qasm}
        if name:
            body["name"] = name
        return self._client._post("/api/circuits/import/qasm", body)


# ── Sampler jobs sub-client ──────────────────────────────────────────────────

# Must match _TERMINAL_STATUSES in backend/app/routes/jobs.py — no shared
# source of truth across the wire, so keep these two in sync by hand.
_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


class SamplerJob:
    """
    Handle for an async sampler job.

    Created by ``client.jobs.submit_direct_sampler(...)`` or
    ``client.jobs.get(id)``. Holds a snapshot plus a reference to the
    client, the same shape as ``Decoder`` — submit returns immediately,
    ``.wait()`` blocks until terminal, ``.stream()`` yields progress,
    ``.cancel()`` is best-effort.

    Example::

        job = client.jobs.submit_direct_sampler({
            "circuit": {...},
            "check_functions": {"anyError": "weight > 0"},
            "num_samples": 10_000,
        })
        job = job.wait()
        print(job.result["decoder_summary"])
    """

    def __init__(self, client: "SnaqcsClient", snapshot: dict) -> None:
        self._client = client
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

    def refresh(self) -> "SamplerJob":
        """Re-fetch the current snapshot from the server."""
        self._snap = self._client._get(f"/api/sampler/jobs/{self.id}")
        return self

    def wait(self, timeout: Optional[float] = None, poll: float = 2.0) -> "SamplerJob":
        """Block until the job reaches a terminal state, then return self.

        Raises ``JobFailedError``/``JobCancelledError`` on those terminal
        states; ``TimeoutError`` if ``timeout`` elapses first.
        """
        start = time.monotonic()
        while self.status not in _TERMINAL_JOB_STATUSES:
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(f"Job {self.id} did not finish within {timeout}s")
            time.sleep(poll)
            self.refresh()
        if self.status == "failed":
            err = self.error or {}
            raise JobFailedError(err.get("message", "unknown error"))
        if self.status == "cancelled":
            raise JobCancelledError(f"Job {self.id} was cancelled")
        return self

    def stream(self, timeout: Optional[float] = None) -> Iterator[dict]:
        """Yield progress snapshots via SSE, updating ``self`` as they arrive.

        ``timeout`` is a total wall-clock budget for the whole call — the
        same semantics as ``wait()``'s ``timeout`` — not a per-request
        socket timeout. It's enforced both while the SSE connection is open
        and after falling back to polling, so a caller who passes
        ``timeout=120`` gets ``TimeoutError`` around 120s of real time
        either way. Pass ``None`` (default) to run until the job reaches a
        terminal state, however long that takes.

        Falls back to polling (2s) when the SSE connection can't be opened
        or drops mid-stream — e.g. behind a reverse proxy not yet
        configured for long-lived streaming responses. A 401 is never
        treated as a transient drop: it's checked explicitly and raises
        ``AuthenticationError`` immediately, matching every other method on
        this client. Likewise, the backend's ``{"error": "not_found"}``
        sentinel (emitted once when the job is soft-deleted mid-stream)
        raises ``JobError`` instead of silently becoming ``self._snap``'s
        new shape and breaking ``.status``/``.id`` on the next access.
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        def _check_deadline():
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"Job {self.id} did not finish within {timeout}s")

        url = f"{self._client.base_url}/api/sampler/jobs/{self.id}/stream"
        try:
            with self._client._session.get(url, stream=True, timeout=timeout) as resp:
                if resp.status_code == 401:
                    raise AuthenticationError(
                        "Authentication failed. Pass api_key= or set SNAQCS_API_KEY env var."
                    )
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
        except requests.RequestException:
            while True:
                _check_deadline()
                self.refresh()
                yield self._snap
                if self.status in _TERMINAL_JOB_STATUSES:
                    return
                time.sleep(2.0)

    def cancel(self, reason: Optional[str] = None) -> "SamplerJob":
        """Request cancellation. Best-effort — the worker checks at the next
        progress tick, so the job may not be ``cancelled`` immediately."""
        body = {"reason": reason} if reason else {}
        self._client._post(f"/api/sampler/jobs/{self.id}/cancel", body)
        return self.refresh()

    def delete(self) -> None:
        """Soft-delete the job."""
        resp = self._client._session.delete(
            f"{self._client.base_url}/api/sampler/jobs/{self.id}", timeout=30
        )
        if resp.status_code == 401:
            raise AuthenticationError(
                "Authentication failed. Pass api_key= or set SNAQCS_API_KEY env var."
            )
        resp.raise_for_status()

    def __repr__(self) -> str:
        return f"SamplerJob(id={self.id!r}, status={self.status!r})"


class Jobs:
    """
    Background sampler jobs sub-client. Access via ``client.jobs``.

    Example::

        job = client.jobs.submit_direct_sampler({...})
        job.wait()
        for j in client.jobs.list(status="running"):
            print(j.id, j.progress)
    """

    def __init__(self, client: "SnaqcsClient") -> None:
        self._client = client

    def submit_direct_sampler(self, request: dict) -> SamplerJob:
        """Submit a direct-sampler job over a protocol graph.

        ``request`` is the same shape ``sample_protocol()`` posts to
        ``/api/protocol/direct_sampler`` — ``{"config": ..., "noise_config":
        ..., "num_samples": ..., "seed": ..., "backend": ...}``. This is the
        protocol-graph sampler; for a single circuit see
        ``submit_circuit_direct_sampler`` instead.
        """
        data = self._client._post(
            "/api/sampler/jobs", {"kind": "direct_sampler", "request": request}
        )
        return self.get(data["job_id"])

    def submit_circuit_direct_sampler(self, request: dict) -> SamplerJob:
        """Submit a single-circuit direct-sampler job.

        ``request`` is the same shape ``sample()`` posts to
        ``/api/direct_sampler`` — ``circuit``, ``check_functions``,
        ``noise_config``, ``num_samples``, ``seed``, ``propagation_backend``,
        ``decoder_backend``/``decoder_config``, etc. (see
        ``SnaqcsClient.sample``'s docstring). Progress streams the same way
        as any other job — ``job.wait()`` or ``job.stream()``.
        """
        data = self._client._post(
            "/api/sampler/jobs",
            {"kind": "circuit_direct_sampler", "request": request},
        )
        return self.get(data["job_id"])

    def get(self, job_id: str) -> SamplerJob:
        """Fetch a job by id."""
        return SamplerJob(self._client, self._client._get(f"/api/sampler/jobs/{job_id}"))

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
        data = self._client._get("/api/sampler/jobs", params=params)
        return [SamplerJob(self._client, item) for item in data["items"]]
