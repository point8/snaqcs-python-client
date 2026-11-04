"""Offline DSS reconstruction helpers against a real server result.

The fixture is a backend run of the GHZ-8 flag protocol with
p_eval=[1e-5, 1e-4]. The helpers mirror the backend's dss_subset_table and
dss_variance (and its reference renewal_bracket); dss_attempt_rates has no
server counterpart since 2026-11-05.

The fixture predates 2026-11-05 and keeps renewal_curve/attempt_rates as the oracle;
the server no longer sends them, so re-capturing it means recomputing them first.
"""

import json
from pathlib import Path

import pytest

from snaqcs import (
    _dss_comb_ratio,
    _dss_counts,
    dss_attempt_rates,
    dss_leading_coefficient,
    dss_p_lower,
    dss_renewal_bracket,
    dss_renewal_p_lower,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def res():
    return json.loads((FIXTURES / "ghz_dss_result.json").read_text())


def test_reconstruction_matches_engine_curve(res):
    assert res["p_curve"]
    for pt in res["p_curve"]:
        assert dss_p_lower(res, pt["p_vec"]) == pytest.approx(pt["p_lower"], rel=1e-9)
    assert dss_leading_coefficient(res) == (
        res["leading_order"],
        pytest.approx(res["leading_coefficient_table"], rel=1e-12),
    )
    for pt in res["p_curve"]:  # same bound as the backend renewal test
        assert (
            abs(dss_renewal_p_lower(res, pt["p_vec"]) - pt["p_lower"])
            <= 3 * pt["sigma_L"] + pt["delta"] + res["truncated_mass"]
        )


def test_scalar_p_scales_the_rate_ratios(res):
    pt = res["p_curve"][-1]
    assert dss_p_lower(res, pt["p"]) == pytest.approx(pt["p_lower"], rel=1e-9)


def test_offline_renewal_bracket_matches_server(res):
    assert res["renewal_curve"]
    for srv in res["renewal_curve"]:
        off = dss_renewal_bracket(res, srv["p_vec"])  # M from res["renewal_max_attempts"]
        for k in ("p_lower", "p_upper", "sigma_L", "p_upper_certified", "U"):
            assert off[k] == pytest.approx(srv[k], rel=1e-9)


def test_renewal_helper_refuses_non_renewal_result(res):
    bad = {**res, "renewal": False, "renewal_note": "check uses visits()"}
    with pytest.raises(ValueError, match="visits"):
        dss_renewal_p_lower(bad, 1e-4)
    with pytest.raises(ValueError, match="visits"):
        dss_renewal_bracket(bad, 1e-4)


def test_helpers_name_the_flag_when_the_tables_are_missing(res):
    # The server leaves the tables out unless include_subset_rates is set.
    bare = {**res, "subset_rates": []}
    bare.pop("attempt_rates", None)
    for helper in (dss_p_lower, dss_renewal_p_lower, dss_renewal_bracket):
        with pytest.raises(ValueError, match="include_subset_rates"):
            helper(bare, 1e-4)
    for helper in (dss_leading_coefficient, dss_attempt_rates):
        with pytest.raises(ValueError, match="include_subset_rates"):
            helper(bare)


def test_attempt_rates_match_the_server_table(res):
    mine, srv = dss_attempt_rates(res), res["attempt_rates"]
    assert len(mine) == len(srv) == 159
    for a, b in zip(mine, srv):
        assert a["steps"] == b["steps"] and a["attempt_weight"] == b["attempt_weight"]
        assert a["f"] == pytest.approx(b["f"], abs=1e-15)
        assert a["r"] == pytest.approx(b["r"], abs=1e-15)


def test_renewal_p_lower_ignores_a_stored_attempt_table(res):
    bare = {k: v for k, v in res.items() if k not in ("attempt_rates", "attempt_sites")}
    for p in (1e-5, 1e-4, 1e-3):
        assert dss_renewal_p_lower(bare, p) == pytest.approx(dss_renewal_p_lower(res, p), rel=1e-12)


def test_attempt_rates_books_a_return_to_r_and_does_not_recurse():
    # prep w=(1,): 10 shots, 4 loop back to prep, 2 fail; nothing below the loop-back is walked
    loop = {"path": [{"node": "prep", "weight": [1]}], "node": "prep", "weight": [0],
            "q_fail_local": 1.0, "branches": {}, "n": 4, "n_discard": 0}
    root = {"path": [], "node": "prep", "weight": [1], "q_fail_local": 0.2,
            "branches": {"prep": 0.4}, "n": 10, "n_discard": 0}
    (a,) = dss_attempt_rates({"renewal": True, "subset_rates": [root, loop]})
    assert a["f"] == pytest.approx(0.2) and a["r"] == pytest.approx(0.4)
    assert a["attempt_weight"] == 1


def test_attempt_rates_refuse_a_non_renewal_result(res):
    with pytest.raises(ValueError, match="visits"):
        dss_attempt_rates({**res, "renewal": False, "renewal_note": "check uses visits()"})


def test_attempt_rows_give_the_tables_leading_coefficient(res):
    # Moved from the backend: attempt-weight-2 rows in Eq.(2) form sum to c2.
    # Repeat paths only enter at order 3.
    order, sites = res["partition_order"], res["fault_sites_by_node"]
    ratios = [r / max(res["p_ref_vec"]) for r in res["p_ref_vec"]]
    c2 = 0.0
    for a in dss_attempt_rates(res):
        if a["attempt_weight"] == 2:
            term = a["f"]
            for s in a["steps"]:
                term *= _dss_comb_ratio(_dss_counts(sites, s["node"], order), s["weight"], ratios)
            c2 += term
    assert c2 == pytest.approx(res["leading_coefficient_table"], rel=1e-12)


def test_renewal_formula_on_a_geometric_toy():
    # One zero-fault subset (A = 1 at any p): f = 0.1 fail, r = 0.5 repeat,
    # so p_L = 0.1 / (1 - 0.5) = 0.2 exactly. Moved from the backend.
    row = {"path": [], "node": "prep", "weight": [0], "q_fail_local": 0.1,
           "branches": {"prep": 0.5}, "n": 10, "n_discard": 0}
    res = {"renewal": True, "subset_rates": [row], "partition_order": ["single"],
           "fault_sites_by_node": {"prep": {"single": 0}}, "p_ref_vec": [1e-3]}
    assert dss_renewal_p_lower(res, 1e-3) == pytest.approx(0.2, rel=1e-15)
