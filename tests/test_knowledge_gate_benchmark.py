"""Gate benchmark scoring and fixed perspective cases."""

from rpbench import knowledge_gate_benchmark


def test_cases_have_expected_labels():
    cases = knowledge_gate_benchmark.CASES

    assert len(cases) == 3
    assert len({case["id"] for case in cases}) == 3

    assert cases[0]["expected"] == ["observable"]
    assert cases[1]["expected"] == ["offscreen"]
    assert cases[2]["expected"] == [
        "observable", "observable", "offscreen"
    ]


def test_scoring_exact_labels(monkeypatch):
    def fake_gate(player_action, npc_text, settings, on_response):
        on_response({
            "content": '{"labels":[]}',
            "reasoning": "",
            "finish_reason": "stop",
            "usage": {"completion_tokens": 20},
        })
        return ["observable"], {}

    monkeypatch.setattr(
        knowledge_gate_benchmark, "evaluate_gate", fake_gate
    )

    result = knowledge_gate_benchmark.run_case(
        knowledge_gate_benchmark.CASES[0], {}
    )

    assert result["valid"]
    assert result["correct"]
    assert result["usage"]["completion_tokens"] == 20


def test_incorrect_labels_are_recorded(monkeypatch):
    def fake_gate(player_action, npc_text, settings, on_response):
        on_response({
            "content": "",
            "reasoning": "",
            "finish_reason": "stop",
            "usage": {},
        })
        return ["observable"], {}

    monkeypatch.setattr(
        knowledge_gate_benchmark, "evaluate_gate", fake_gate
    )

    result = knowledge_gate_benchmark.run_case(
        knowledge_gate_benchmark.CASES[1], {}
    )

    assert result["valid"]
    assert not result["correct"]
    assert result["labels"] == ["observable"]


def test_invalid_generation_recorded(monkeypatch):
    def fake_gate(player_action, npc_text, settings, on_response):
        on_response({
            "content": "",
            "reasoning": "Repeated analysis",
            "finish_reason": "length",
            "usage": {"completion_tokens": 8192},
        })
        raise ValueError("Knowledge gate exhausted token budget")

    monkeypatch.setattr(
        knowledge_gate_benchmark, "evaluate_gate", fake_gate
    )

    result = knowledge_gate_benchmark.run_case(
        knowledge_gate_benchmark.CASES[0], {}
    )

    assert not result["valid"]
    assert not result["correct"]
    assert result["reasoning_only_truncation"]
