"""SNAQCS Python client — single entry point for all API operations."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class AuthenticationError(Exception):
    """Raised when the API key is missing, invalid, or revoked."""


class ServerUnavailableError(Exception):
    """Raised when the SNAQCS server cannot be reached."""


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
    is_correctable: bool
    initial_error: str
    final_error: str
    final_syndrome: dict
    final_correction: dict
    code_parameters: dict
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
        # base URL defaults to SNAQCS_API_URL env var, then http://localhost:6090

    Quick decode::

        result = client.decode("XIIII", stabilizers=["+XZZXI", ...], num_qubits=5)

    Code-specific decoder (Stim-like interface, reuses connection)::

        decoder = client.decoder(stabilizers=["+XZZXI", ...], num_qubits=5)
        result = decoder.decode("XIIII")
        results = decoder.decode_batch(["XIIII", "IXIII"])

    Fault propagation::

        result = client.propagate(circuit={...}, faults=[...], code_config={...})
        print(result.is_correctable, result.final_error)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = (
            (base_url or os.environ.get("SNAQCS_API_URL") or "http://localhost:6090")
            .rstrip("/")
        )
        key = api_key or os.environ.get("SNAQCS_API_KEY")
        self._session = _make_session(key)

    def _post(self, path: str, body: dict) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(url, json=body, timeout=60)
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
        return resp.json()

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
        return resp.json()

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
        code_config: dict,
        decoder_type: str = "table",
        return_intermediate: bool = True,
    ) -> PropagationResult:
        """
        Propagate faults through a circuit and analyse correctability.

        ``faults`` is a list of ``{"location": N, "qubit": Q, "pauli": "X"}`` dicts.
        ``location=N`` means the error appears after gate N; ``location=-1`` is a prep fault.
        """
        data = self._post("/api/propagate_with_correctability", {
            "circuit": circuit,
            "faults": faults,
            "code_config": code_config,
            "decoder_type": decoder_type,
            "return_intermediate": return_intermediate,
        })
        return PropagationResult(
            is_correctable=data["is_correctable"],
            initial_error=data["initial_error"],
            final_error=data["final_error"],
            final_syndrome=data["final_syndrome"],
            final_correction=data["final_correction"],
            code_parameters=data["code_parameters"],
            intermediate_steps=data.get("intermediate_steps", []),
        )

    def enumerate_faults(
        self,
        circuit: dict,
        max_fault_weight: int,
        code_config: Optional[dict] = None,
    ) -> dict:
        """Enumerate all fault configurations up to a given weight."""
        return self._post("/api/enumerate_faults", {
            "circuit": circuit,
            "max_fault_weight": max_fault_weight,
            "code_config": code_config,
        })

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        circuit: dict,
        code_config: dict,
        noise_config: Optional[dict] = None,
        num_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> dict:
        """Monte Carlo fault sampling for a single circuit with depolarizing noise."""
        return self._post("/api/direct_sampler", {
            "circuit": circuit,
            "code_config": code_config,
            "noise_config": noise_config or {},
            "num_samples": num_samples,
            "seed": seed,
        })

    def sample_protocol(
        self,
        config: dict,
        noise_config: Optional[dict] = None,
        num_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> dict:
        """Monte Carlo fault sampling for a multi-circuit protocol."""
        return self._post("/api/protocol/direct_sampler", {
            "config": config,
            "noise_config": noise_config or {},
            "num_samples": num_samples,
            "seed": seed,
        })

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
        code_config: Optional[dict] = None,
        return_intermediate: bool = False,
    ) -> dict:
        """Propagate multiple simultaneous faults and return the combined error."""
        return self._post("/api/multiple_faults", {
            "circuit": circuit,
            "faults": faults,
            "code_config": code_config,
            "return_intermediate": return_intermediate,
        })

    def enumerate_single_faults(
        self,
        circuit: dict,
        code_config: Optional[dict] = None,
        fault_types: Optional[list] = None,
        include_idle: bool = False,
    ) -> dict:
        """Enumerate every single-fault location in the circuit."""
        return self._post("/api/enumerate_single_faults", {
            "circuit": circuit,
            "code_config": code_config,
            "fault_types": fault_types or ["X", "Y", "Z"],
            "include_idle": include_idle,
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
        return self._post("/api/wilson_ci", {
            "num_success": num_success,
            "num_total": num_total,
            "ci_z": ci_z,
        })

    # ── Protocol ──────────────────────────────────────────────────────────────

    def simulate_protocol(
        self,
        config: dict,
        faults: list,
    ) -> dict:
        """Simulate a specific fault scenario through a multi-circuit protocol."""
        return self._post("/api/protocol/simulate", {
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
