import logging
from types import SimpleNamespace

import pytest

from explanation import groq_explainer as ge

RESPONSE = """EXPLANATION:
Strong overlap in Python and SQL.

STRENGTHS:
- Python
- SQL

GAPS:
- No cloud experience

SUGGESTIONS:
- Add AWS projects
"""

BREAKDOWN = {"hybrid_score": 0.7, "matched_skills": ["python"], "missing_skills": ["aws"]}


class NotFound(Exception):
    status_code = 404


class FakeClient:
    """Stands in for groq.Groq: records calls, fails for chosen models."""

    def __init__(self, live_models=None, list_error=None, failing=()):
        self.list_calls = 0
        self.used = []
        self._live, self._list_error, self._failing = live_models or [], list_error, set(failing)
        self.models = SimpleNamespace(list=self._list)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _list(self):
        self.list_calls += 1
        if self._list_error:
            raise self._list_error
        return SimpleNamespace(data=[SimpleNamespace(id=m, active=True) for m in self._live])

    def _create(self, model, **kw):
        self.used.append(model)
        if model in self._failing:
            raise NotFound(f"Error code: 404 - {{'error': {{'code': 'model_not_found', 'model': '{model}'}}}}")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=RESPONSE))])


@pytest.fixture(autouse=True)
def fresh_cache():
    ge.reset_model_cache()
    yield
    ge.reset_model_cache()


def run(client):
    return ge.generate_match_explanation_groq(["chunk"], "job text", BREAKDOWN, client=client)


def test_rank_models_skips_non_chat_and_prefers_general_llama():
    live = ["whisper-large-v3", "playai-tts", "llama-guard-4-12b", "openai/gpt-oss-20b",
            "llama-3.1-8b-instant", "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile"]
    assert ge.rank_models(live) == [
        "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant", "openai/gpt-oss-20b"]


def test_uses_live_model_list_not_hardcoded_name():
    client = FakeClient(live_models=["llama-4-future-instruct", "whisper-large-v3"])
    result = run(client)
    assert client.used == ["llama-4-future-instruct"]
    assert result["explanation"] == "Strong overlap in Python and SQL."
    assert result["strengths"] == ["Python", "SQL"] and "error" not in result


def test_models_endpoint_called_once_across_requests():
    client = FakeClient(live_models=["llama-3.3-70b-versatile"])
    run(client); run(client); run(client)
    assert client.list_calls == 1


def test_falls_back_to_hardcoded_models_when_list_fails():
    client = FakeClient(list_error=RuntimeError("401 bad key"), failing={ge.FALLBACK_MODELS[0]})
    result = run(client)
    assert client.used == ge.FALLBACK_MODELS[:2]  # first failed, second worked
    assert "error" not in result


def test_next_candidate_is_tried_when_a_model_is_missing():
    client = FakeClient(live_models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                        failing={"llama-3.3-70b-versatile"})
    assert "error" not in run(client)
    assert client.used == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def test_model_not_found_triggers_re_resolution():
    client = FakeClient(live_models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                        failing={"llama-3.3-70b-versatile"})
    run(client)
    run(client)
    assert client.list_calls == 2


def test_total_failure_shows_friendly_message_and_logs_real_error(caplog):
    client = FakeClient(live_models=["a-model"], failing={"a-model"})
    with caplog.at_level(logging.ERROR):
        result = run(client)
    assert result["error"] is True
    assert result["explanation"] == ge.FRIENDLY_ERROR
    assert "Error code" not in result["explanation"] and "404" not in result["explanation"]
    assert result["strengths"] == result["gaps"] == result["suggestions"] == []
    assert "model_not_found" in caplog.text  # real error preserved for debugging


def test_no_api_key_uses_offline_fallback(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = ge.generate_match_explanation_groq(["c"], "job", BREAKDOWN)
    assert "error" not in result and result["explanation"]


def test_auth_failure_does_not_try_every_model():
    class AuthError(Exception):
        status_code = 401

    client = FakeClient(live_models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    client._create = lambda model, **kw: (client.used.append(model), (_ for _ in ()).throw(AuthError("bad key")))[1]
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=client._create))
    assert run(client)["error"] is True
    assert len(client.used) == 1
