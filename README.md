# snaQSs Python Client

Python client for the snaQCs quantum error correction API. Authenticate once
with your API key, get typed result objects, and use all API endpoints from a
single entry point.

All examples can be run after setting `SNAQCS_API_KEY` and `SNAQCS_API_URL`.
See Auth below.

```bash
git clone https://github.com/point8/snaqcs-python-client
cd snaqcs-python-client
uv run example_fault_analysis.py
```

## Server compatibility

0.3.0 and later talk to the versioned `/api/v1` server API and need a snaQCs
server from the 2026-10-06 URL rename onwards. Older servers (unversioned
`/api/...` paths) need client 0.2.x; there is no version that works with both.

## Auth

The client requires an API key to connect to the snaQCs server:

```python
# Pass directly
client = SnaqcsClient(api_key="snaqcs_...", base_url="https://snaqcs.point8.cloud")
```

Or export environment variables before running:

```bash
export SNAQCS_API_KEY=snaqcs_...
export SNAQCS_API_URL=https://snaqcs.point8.cloud
```

**For contributors:** If you are running the snaQCs backend locally for
development (`just start-backend` → `http://localhost:6090`), you can connect
without a key — the dev server runs with authentication disabled. `client =
SnaqcsClient()` will default to `http://localhost:6090` when no `SNAQCS_API_URL` is
set.

## Declaring the target state

A protocol judged by a decoder says which logical state it prepares in
`target_state`, part of the protocol config. Each entry is a signed logical of
the code: `Z<i>` is `logical_z[i]`, `X<i>` is `logical_x[i]`.

| State | `target_state` |
|---|---|
| \|0>_L | unset (the default), or `["+Z0"]` |
| \|1>_L | `["-Z0"]` |
| \|+>_L | `["+X0"]` |
| \|->_L | `["-X0"]` |
| logical Bell pair, ZZ parity | `["+Z0Z1"]` |

```python
config = {**protocol, "target_state": ["+X0"]}
result = client.sample_protocol(config=config, noise_config=noise, num_samples=10_000)
result["verdict_target"]
# {'target_state': ['+X0'], 'readout_basis': 'X', 'error_sector_judged': 'Z'}
```

The readout backends (tableau, clifft, SUSA) read the data out in the target's
basis. A protocol that does not prepare the declared state is refused before
sampling.

**Comparing bases.** A Z-basis target (`|0>`, `|1>`) gives the rate of logical X
flips. An X-basis target (`|+>`, `|->`) gives the rate of logical Z flips. They
are two different quantities from two runs, not one p_L. Their mean is the
basis-averaged rate, and their sum bounds the rate of any logical error.

## Non-Clifford circuits on clifft

`client.sample(...)` runs on the Pauli frame unless you pass `backend="clifft"`.
clifft simulates non-Clifford circuits (T, CCX) exactly. `m_i` still means a
flip: the outcome XOR the noiseless outcome, which the server certifies exactly
for every bit a check reads (`reference_check` in the result). A check that reads
`final_error` or `weight`, the decoder, correlated noise and sample details are
refused on clifft, and the weight histogram comes back as `None`. Each check
carries a 3-sigma Wilson interval (`ci_low`, `ci_high`), and the result names
the `seed` it used, so a run can be replayed.

This is direct Monte Carlo (at most 1e6 shots), so p_L below about 1e-5 is out
of reach here; for low p use the subset sampler (SUSA).

## Development

```bash
uv run pytest tests/ -v
```
Integration tests against a local dev server are skipped automatically when the
server is not running.


## Acknowledgements

We acknowledge support for snaQCs development from the Bundesministeriums für
Forschung, Technologie und Raumfahrt (BMFTR) with the project snaqcs2025 -
Scalable near-term algorithms for quantum computing systems.


## License

MIT — see [LICENSE](LICENSE) for details.
