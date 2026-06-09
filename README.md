# snaQSs Python Client

Python client for the snaQCs quantum error correction API. Authenticate once
with your API key, get typed result objects, and use all API endpoints from a
single entry point.

## Examples

Two runnable examples are included:

- **`example_mc_sampling.py`** — Monte Carlo logical error rate estimation.
Sweeps three noise levels, runs 10,000 shots each, and prints p_L with a 2σ
Wilson confidence interval.
- **`example_fault_analysis.py`** — Deterministic single-fault analysis.
Enumerates every fault location in the circuit and reports which gate positions
produce uncorrectable logical errors.

Both examples use the same [[7,1,3]] Steane code encoder circuit and can be run
after setting `SNAQCS_API_KEY` and `SNAQCS_API_URL`. See Auth below.

```bash
git clone https://github.com/point8/snaqcs-python-client
cd snaqcs-python-client
uv run example_mc_sampling.py
uv run example_fault_analysis.py
```

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
development (`make start-backend` → `http://localhost:6090`), you can connect
without a key — the dev server runs with authentication disabled. `client =
SnaqcsClient()` will default to `http://localhost:6090` when no `SNAQCS_API_URL` is
set.

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

