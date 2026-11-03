"""Offline DSS reconstruction helpers against a real server result.

The fixture is a backend run of the GHZ-8 flag protocol with 
p_eval=[1e-5, 1e-4]. The helpers mirror the backend's dss_subset_table
/ dss_renewal / dss_variance; these tests are what keep the two from
drifting apart.
"""

import json
from pathlib import Path

import pytest

from snaqcs import (
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
