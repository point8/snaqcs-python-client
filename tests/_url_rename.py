"""The 2026-10-06 API URL rename as data: old full path -> new full path.

Single source of truth for the rename (docs/API_URL_NAMING.md §5, §10.1, a
local gitignored plan). Every current path is a key, including the ones whose
segment below the prefix is unchanged, because the prefix itself moves from
``/api`` to ``/api/v1``.

Map rule: every templated key has an untemplated prefix key. Callers build
templated paths with f-strings or ``${id}``, so ``{job_id}`` never appears
literally in code; a stale call site is only findable through its untemplated
prefix (``/api/sampler/jobs``).

``rewrite`` is the one rewrite used by the stale-literal test and by the
migration of test and client literals: longest key first, one pass, with a
``(?![a-z_])`` right boundary so ``/api/decode`` does not hit ``/api/decoder``.
"""

import re

RENAMES: dict[str, str] = {
    # system / users / admin — unchanged below the prefix
    "/api/health": "/api/v1/health",
    "/api/version": "/api/v1/version",
    "/api/me": "/api/v1/me",
    "/api/me/keys": "/api/v1/me/keys",
    "/api/me/keys/{key_id}": "/api/v1/me/keys/{key_id}",
    "/api/admin/library/gc": "/api/v1/admin/library/gc",
    "/api/admin/library/gc/purge": "/api/v1/admin/library/gc/purge",
    # circuits — unchanged below the prefix
    "/api/circuits": "/api/v1/circuits",
    "/api/circuits/id": "/api/v1/circuits/id",
    "/api/circuits/id/{circuit_id}": "/api/v1/circuits/id/{circuit_id}",
    "/api/circuits/id/{circuit_id}/versions/{version}": (
        "/api/v1/circuits/id/{circuit_id}/versions/{version}"
    ),
    "/api/circuits/{name}": "/api/v1/circuits/{name}",
    "/api/circuits/{name}/visibility": "/api/v1/circuits/{name}/visibility",
    "/api/circuits/{name}/owner": "/api/v1/circuits/{name}/owner",
    "/api/circuits/{name}/export/qasm": "/api/v1/circuits/{name}/export/qasm",
    "/api/circuits/{name}/export/json": "/api/v1/circuits/{name}/export/json",
    "/api/circuits/import/qasm": "/api/v1/circuits/import/qasm",
    # noise_models
    "/api/noise/library": "/api/v1/noise_models",
    "/api/noise/library/id": "/api/v1/noise_models/id",
    "/api/noise/library/id/{noise_id}": "/api/v1/noise_models/id/{noise_id}",
    "/api/noise/library/id/{noise_id}/versions": "/api/v1/noise_models/id/{noise_id}/versions",
    "/api/noise/library/id/{noise_id}/versions/{version}": (
        "/api/v1/noise_models/id/{noise_id}/versions/{version}"
    ),
    "/api/noise/library/{name}": "/api/v1/noise_models/{name}",
    "/api/noise/library/{name}/visibility": "/api/v1/noise_models/{name}/visibility",
    "/api/noise/library/{name}/owner": "/api/v1/noise_models/{name}/owner",
    "/api/noise/profile/validate": "/api/v1/noise_models/profile/validate",
    # protocols
    "/api/protocols/library": "/api/v1/protocols",
    "/api/protocols/library/id": "/api/v1/protocols/id",
    "/api/protocols/library/id/{protocol_id}": "/api/v1/protocols/id/{protocol_id}",
    "/api/protocols/library/{name}": "/api/v1/protocols/{name}",
    "/api/protocols/library/{name}/visibility": "/api/v1/protocols/{name}/visibility",
    "/api/protocols/library/{name}/owner": "/api/v1/protocols/{name}/owner",
    "/api/protocols/library/{name}/hydrated": "/api/v1/protocols/{name}/hydrated",
    # analysis
    "/api/propagate": "/api/v1/analysis/circuit/propagate",
    "/api/multiple_faults": "/api/v1/analysis/circuit/propagate_multi",
    "/api/enumerate_faults": "/api/v1/analysis/circuit/enumerate_faults",
    "/api/enumerate_pauli_faults": "/api/v1/analysis/circuit/enumerate_pauli_faults",
    "/api/circuit/valid_fault_ticks": "/api/v1/analysis/circuit/valid_fault_ticks",
    "/api/wilson_ci": "/api/v1/analysis/wilson_ci",
    "/api/protocol/propagate": "/api/v1/analysis/protocol/propagate",
    "/api/protocol/replay": "/api/v1/analysis/protocol/replay",
    "/api/protocol/replayability": "/api/v1/analysis/protocol/replayability",
    "/api/protocols/hydrate": "/api/v1/analysis/protocol/hydrate",
    # sampling
    "/api/circuit/direct_sampler": "/api/v1/sampling/circuit/direct",
    "/api/protocol/direct_sampler": "/api/v1/sampling/protocol/direct",
    "/api/protocol/subset_sampler": "/api/v1/sampling/protocol/subset",
    "/api/sampler/jobs": "/api/v1/sampling/jobs",
    "/api/sampler/jobs/{job_id}": "/api/v1/sampling/jobs/{job_id}",
    "/api/sampler/jobs/{job_id}/checkpoint": "/api/v1/sampling/jobs/{job_id}/checkpoint",
    "/api/sampler/jobs/{job_id}/cancel": "/api/v1/sampling/jobs/{job_id}/cancel",
    "/api/sampler/jobs/{job_id}/stream": "/api/v1/sampling/jobs/{job_id}/stream",
    "/api/sampler/jobs/stream/me": "/api/v1/sampling/jobs/stream/me",
    "/api/sampler/jobs/bulk-delete": "/api/v1/sampling/jobs/bulk_delete",
    "/api/sampler/workers": "/api/v1/sampling/workers",
    "/api/sampler/workers/{worker_id}": "/api/v1/sampling/workers/{worker_id}",
    # decoder
    "/api/decode": "/api/v1/decoder/decode",
    "/api/decode/batch": "/api/v1/decoder/decode_batch",
    "/api/decode/info": "/api/v1/decoder/info",
    "/api/syndrome": "/api/v1/decoder/syndrome",
    # visualization
    "/api/visualize/circuit/data": "/api/v1/visualization/circuit",
    "/api/visualize/protocol_graph/data": "/api/v1/visualization/protocol_graph",
    "/api/compile/typst": "/api/v1/visualization/typst",
    # import — unchanged below the prefix
    "/api/import/qasm2": "/api/v1/import/qasm2",
    "/api/import/qasm3": "/api/v1/import/qasm3",
    "/api/import/braket": "/api/v1/import/braket",
    "/api/import/stim": "/api/v1/import/stim",
}

# Old keys that are not routes themselves, only the untemplated stems of
# templated routes (see the map rule above).
UNTEMPLATED_PREFIXES = {"/api/circuits/id", "/api/noise/library/id", "/api/protocols/library/id"}

_UNTEMPLATED_KEYS = sorted((k for k in RENAMES if "{" not in k), key=len, reverse=True)
OLD_PATH_RE = re.compile("(" + "|".join(re.escape(k) for k in _UNTEMPLATED_KEYS) + r")(?![a-z_])")


def rewrite(text: str) -> str:
    """Rewrite every old URL literal in ``text`` to its new path, in one pass."""
    return OLD_PATH_RE.sub(lambda m: RENAMES[m.group(1)], text)
