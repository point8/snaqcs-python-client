"""Tests for the shared _Transport seam.

_Transport is the single place that translates ConnectionError ->
ServerUnavailableError and HTTP 401 -> AuthenticationError. SnaqcsClient,
Circuits, and Jobs/SamplerJob all send requests through the same instance.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from snaqcs import (
    _Transport,
    AuthenticationError,
    ServerUnavailableError,
    UnexpectedResponseError,
)


def _mock_resp(data=None, status=200, url="http://localhost:6090/api/x", text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.url = url
    resp.text = text
    resp.headers = {"Content-Type": "application/json"}
    resp.raise_for_status = MagicMock()
    if data is not None:
        resp.json.return_value = data
    else:
        resp.json.side_effect = requests.exceptions.JSONDecodeError("", "", 0)
    return resp


# ── Session setup ─────────────────────────────────────────────────────────────

def test_no_api_key_means_no_auth_header():
    t = _Transport("http://localhost:6090", api_key=None, timeout=60.0)
    assert "Authorization" not in t._session.headers


def test_api_key_sets_bearer_auth_header():
    t = _Transport("http://localhost:6090", api_key="snaqcs_x", timeout=60.0)
    assert t._session.headers["Authorization"] == "Bearer snaqcs_x"


# ── send() ────────────────────────────────────────────────────────────────────

def test_send_builds_full_url_from_base_and_path():
    t = _Transport("http://localhost:6090", api_key=None, timeout=60.0)
    with patch.object(t._session, "get", return_value=_mock_resp({})) as mock_get:
        t.send("GET", "/api/health")
    assert mock_get.call_args[0] == ("http://localhost:6090/api/health",)


def test_send_connection_error_raises_server_unavailable_error():
    t = _Transport("http://localhost:6090", api_key=None, timeout=60.0)
    with patch.object(t._session, "get", side_effect=requests.exceptions.ConnectionError):
        with pytest.raises(ServerUnavailableError):
            t.send("GET", "/api/health")


def test_send_401_raises_authentication_error():
    t = _Transport("http://localhost:6090", api_key="snaqcs_bad", timeout=60.0)
    with patch.object(t._session, "get", return_value=_mock_resp(status=401)):
        with pytest.raises(AuthenticationError):
            t.send("GET", "/api/health")


def test_send_401_with_translate_auth_errors_false_returns_response():
    t = _Transport("http://localhost:6090", api_key=None, timeout=60.0)
    resp = _mock_resp(status=401)
    with patch.object(t._session, "get", return_value=resp):
        result = t.send("GET", "/api/health", translate_auth_errors=False)
    assert result is resp


def test_send_defaults_to_transport_timeout():
    t = _Transport("http://localhost:6090", api_key=None, timeout=42.0)
    with patch.object(t._session, "get", return_value=_mock_resp({})) as mock_get:
        t.send("GET", "/api/health")
    assert mock_get.call_args[1]["timeout"] == 42.0


def test_send_explicit_timeout_overrides_default():
    t = _Transport("http://localhost:6090", api_key=None, timeout=42.0)
    with patch.object(t._session, "get", return_value=_mock_resp({})) as mock_get:
        t.send("GET", "/api/health", timeout=5.0)
    assert mock_get.call_args[1]["timeout"] == 5.0


def test_send_dispatches_post_to_session_post():
    t = _Transport("http://localhost:6090", api_key=None, timeout=60.0)
    with patch.object(t._session, "post", return_value=_mock_resp({})) as mock_post:
        t.send("POST", "/api/x", json={"a": 1})
    assert mock_post.call_args[0] == ("http://localhost:6090/api/x",)
    assert mock_post.call_args[1]["json"] == {"a": 1}


def test_send_dispatches_delete_to_session_delete():
    t = _Transport("http://localhost:6090", api_key=None, timeout=60.0)
    with patch.object(t._session, "delete", return_value=_mock_resp({})) as mock_delete:
        t.send("DELETE", "/api/x")
    assert mock_delete.call_args[0] == ("http://localhost:6090/api/x",)


# ── send_json() ───────────────────────────────────────────────────────────────

def test_send_json_returns_parsed_body():
    t = _Transport("http://localhost:6090", api_key=None, timeout=60.0)
    with patch.object(t._session, "post", return_value=_mock_resp({"a": 1})):
        result = t.send_json("POST", "/api/x", json={})
    assert result == {"a": 1}


def test_send_json_oauth2_redirect_raises_authentication_error():
    t = _Transport("http://localhost:6090", api_key=None, timeout=60.0)
    resp = _mock_resp({}, url="http://localhost:6090/oauth2/sign_in")
    with patch.object(t._session, "get", return_value=resp):
        with pytest.raises(AuthenticationError):
            t.send_json("GET", "/api/x")


def test_send_json_non_json_body_raises_unexpected_response_error():
    t = _Transport("http://localhost:6090", api_key=None, timeout=60.0)
    resp = _mock_resp(data=None, text="<html>not json</html>")
    with patch.object(t._session, "get", return_value=resp):
        with pytest.raises(UnexpectedResponseError):
            t.send_json("GET", "/api/x")
