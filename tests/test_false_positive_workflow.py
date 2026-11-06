import json
from pathlib import Path

import pytest

import rpbench.false_positive_workflow as workflow
from rpbench.state import build_ground_truth


def make_run(root):
    root.mkdir(parents=True)

    player = {
        "name": "Alex",
        "main_goal": "Find the key",
        "subgoals": ["Investigate the archive"],
    }

    definition = {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "requires": [],
                "criteria": [{
                    "id": "evidence",
                    "description": "Find evidence",
                    "evidence_contains": "evidence",
                }],
            },
        },
    }

    manifest = {
        "type": "goal_driven_selfplay",
        "status": "complete",
        "runtime": {"goal_criteria": definition},
    }

    for name, data in (
        ("manifest.json", manifest),
        ("ground_truth.json", build_ground_truth(player, [])),
    ):
        (root / name).write_text(json.dumps(data))

    (root / "events.jsonl").write_text("")
    (root / "observations.jsonl").write_text("")

    return root


def fake_candidates(*args):
    return [{"turn": 1, "goal_id": "subgoal-01"}]


def fake_audit(statuses, calls):
    def impl(
        source, output, manifest, definition,
        observations, candidate, model, base_url,
        max_tokens,
    ):
        calls.append((str(source), model))
        status = statuses.pop(0)
        criterion_status = {
            "valid": "satisfied",
            "false_positive": "not_satisfied",
            "uncertain": "uncertain",
        }[status]
        return {
            "classification": status,
            "verdict": {
                "criteria": [{
                    "criterion_id": "evidence",
                    "status": criterion_status,
                }],
            },
        }
    return impl


def test_dry_inspection_has_no_completions(tmp_path):
    root = make_run(tmp_path / "run")
    _, _, _, candidates = workflow.inspect_run(root)
    assert candidates == []


def test_valid_run_does_not_branch(tmp_path, monkeypatch):
    source = make_run(tmp_path / "run")
    monkeypatch.setattr(
        workflow, "find_completions", fake_candidates
    )

    calls = []
    monkeypatch.setattr(
        workflow, "audit_one",
        fake_audit(["valid"], calls),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected repair")

    monkeypatch.setattr(
        workflow, "prepare_branch", forbidden
    )
    monkeypatch.setattr(
        workflow, "resume_simulation", forbidden
    )

    result = workflow.run_workflow(
        source, tmp_path / "workspace",
    )

    assert result["status"] == "audited"
    assert result["repairs"] == 0
    assert result["final_run"] == str(source)
    assert len(calls) == 1


def test_single_judge_cannot_trigger_repair(
    tmp_path, monkeypatch
):
    source = make_run(tmp_path / "run")
    monkeypatch.setattr(
        workflow, "find_completions", fake_candidates
    )
    monkeypatch.setattr(
        workflow, "audit_one",
        fake_audit(["false_positive"], []),
    )

    result = workflow.run_workflow(
        source, tmp_path / "workspace",
    )

    assert result["status"] == "needs_confirmation"
    assert result["repairs"] == 0


@pytest.mark.parametrize(
    "confirmation, expected",
    [
        ("valid", "disputed"),
        ("uncertain", "uncertain"),
    ],
)
def test_disagreement_blocks_repair(
    tmp_path, monkeypatch, confirmation, expected
):
    source = make_run(tmp_path / "run")
    monkeypatch.setattr(
        workflow, "find_completions", fake_candidates
    )
    monkeypatch.setattr(
        workflow, "audit_one",
        fake_audit(
            ["false_positive", confirmation], []
        ),
    )

    result = workflow.run_workflow(
        source,
        tmp_path / "workspace",
        confirmation_model="independent-judge",
    )

    assert result["status"] == expected
    assert result["repairs"] == 0


def test_confirmed_fp_branches_resumes_and_reaudits(
    tmp_path, monkeypatch
):
    source = make_run(tmp_path / "run")
    monkeypatch.setattr(
        workflow, "find_completions", fake_candidates
    )

    calls = []
    monkeypatch.setattr(
        workflow, "audit_one",
        fake_audit(
            ["false_positive", "false_positive", "valid"],
            calls,
        ),
    )

    def make_branch(source_root, destination, audit_root):
        make_run(Path(destination))
        manifest = json.loads(
            (destination / "manifest.json").read_text()
        )
        manifest["status"] = "failed"
        (destination / "manifest.json").write_text(
            json.dumps(manifest)
        )
        return {"synthetic": False}

    def resume_branch(destination):
        manifest_file = destination / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["status"] = "complete"
        manifest_file.write_text(json.dumps(manifest))

    monkeypatch.setattr(
        workflow, "prepare_branch", make_branch
    )
    monkeypatch.setattr(
        workflow, "resume_simulation", resume_branch
    )

    result = workflow.run_workflow(
        source,
        tmp_path / "workspace",
        confirmation_model="independent-judge",
    )

    assert result["status"] == "audited"
    assert result["repairs"] == 1
    assert len(result["rounds"]) == 2
    assert Path(result["final_run"]).is_dir()
    assert result["final_run"] != str(source)
    assert len(calls) == 3

    # Original run remains unchanged.
    assert json.loads(
        (source / "manifest.json").read_text()
    )["status"] == "complete"


def test_repair_limit_is_enforced(tmp_path, monkeypatch):
    source = make_run(tmp_path / "run")
    monkeypatch.setattr(
        workflow, "find_completions", fake_candidates
    )
    monkeypatch.setattr(
        workflow, "audit_one",
        fake_audit(
            ["false_positive"] * 6,
            [],
        ),
    )

    def make_branch(source_root, destination, audit_root):
        make_run(Path(destination))
        manifest = json.loads(
            (destination / "manifest.json").read_text()
        )
        manifest["status"] = "failed"
        (destination / "manifest.json").write_text(
            json.dumps(manifest)
        )
        return {"synthetic": False}

    def resume_branch(destination):
        file = destination / "manifest.json"
        data = json.loads(file.read_text())
        data["status"] = "complete"
        file.write_text(json.dumps(data))

    monkeypatch.setattr(
        workflow, "prepare_branch", make_branch
    )
    monkeypatch.setattr(
        workflow, "resume_simulation", resume_branch
    )

    result = workflow.run_workflow(
        source,
        tmp_path / "workspace",
        confirmation_model="independent-judge",
        max_repairs=2,
    )

    assert result["status"] == "repair_limit"
    assert result["repairs"] == 2
    assert len(result["rounds"]) == 3



def test_different_confirmation_model_required(
    tmp_path,
):
    source = make_run(tmp_path / "run")

    with pytest.raises(ValueError, match="different judge"):
        workflow.run_workflow(
            source,
            tmp_path / "workspace",
            model="same-model",
            confirmation_model="same-model",
        )

    assert not (tmp_path / "workspace").exists()


def test_disagreement_on_rejected_criteria_blocks_branch(
    tmp_path, monkeypatch,
):
    source = make_run(tmp_path / "run")

    monkeypatch.setattr(
        workflow, "find_completions", fake_candidates
    )

    responses = [
        {
            "classification": "false_positive",
            "verdict": {
                "criteria": [{
                    "criterion_id": "evidence",
                    "status": "not_satisfied",
                }],
            },
        },
        {
            "classification": "false_positive",
            "verdict": {
                "criteria": [{
                    "criterion_id": "other",
                    "status": "not_satisfied",
                }],
            },
        },
    ]

    monkeypatch.setattr(
        workflow,
        "audit_one",
        lambda *args: responses.pop(0),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Disputed evidence must not create a branch"
        )

    monkeypatch.setattr(
        workflow, "prepare_branch", forbidden
    )

    result = workflow.run_workflow(
        source,
        tmp_path / "workspace",
        confirmation_model="independent-judge",
    )

    assert result["status"] == "disputed"
    assert result["repairs"] == 0
    assert result["rounds"][0]["audits"][0][
        "rejected_criteria"
    ] == ["evidence"]
    assert result["rounds"][0]["audits"][0][
        "confirmation_rejected_criteria"
    ] == ["other"]
