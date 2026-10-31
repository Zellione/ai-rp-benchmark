import hashlib
import json
from pathlib import Path

import pytest

from rpbench import simulation_matrix


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def make_job(tmp_path):
    sources = {}

    for kind in ("scenario", "player", "config"):
        file = tmp_path / f"{kind}.json"
        file.write_text(f'{{"kind":"{kind}"}}')
        sources[kind] = hashlib.sha256(file.read_bytes()).hexdigest()

    return {
        "key": "model__de__scene__baseline__test",
        "identity": {
            "turns": 3,
            "sources": sources,
        },
    }


def make_archived_job(tmp_path, status="failed"):
    job = make_job(tmp_path)
    root = tmp_path / "output" / job["key"]
    run = root / "runs" / "archived-selfplay"

    write_json(root / "job.json", {
        "status": status,
        "identity": job["identity"],
    })

    write_json(run / "manifest.json", {
        "type": "goal_driven_selfplay",
        "status": status,
        "planned_turns": 3,
        "completed_turns": 2,
        "scenario_sha256": job["identity"]["sources"]["scenario"],
        "player_sha256": job["identity"]["sources"]["player"],
        "config_sha256": job["identity"]["sources"]["config"],
    })

    return job, root, run


def test_failed_job_requires_explicit_resume(tmp_path):
    job, root, run = make_archived_job(tmp_path)

    with pytest.raises(ValueError, match="resume-failed"):
        simulation_matrix.execute(
            [job], root.parent,
        )


def test_resume_reuses_existing_archive(tmp_path, monkeypatch):
    job, root, run = make_archived_job(tmp_path)
    called = []

    def fake_resume(run_dir):
        called.append(Path(run_dir))
        manifest_path = Path(run_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "complete"
        manifest["completed_turns"] = 3
        write_json(manifest_path, manifest)

    monkeypatch.setattr(
        "rpbench.resume.resume_simulation",
        fake_resume,
    )

    simulation_matrix.execute(
        [job], root.parent,
        resume_failed=True,
    )

    assert called == [run.resolve()]
    state = json.loads((root / "job.json").read_text())
    assert state["status"] == "complete"
    assert Path(state["run_dir"]) == run.resolve()


def test_resume_rejects_modified_archive(tmp_path, monkeypatch):
    job, root, run = make_archived_job(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scenario_sha256"] = "0" * 64
    write_json(manifest_path, manifest)

    called = []
    monkeypatch.setattr(
        "rpbench.resume.resume_simulation",
        lambda _: called.append(True),
    )

    with pytest.raises(ValueError, match="Archived input mismatch"):
        simulation_matrix.execute(
            [job], root.parent,
            resume_failed=True,
        )

    assert not called


def test_dry_run_never_resumes(tmp_path, monkeypatch):
    job, root, run = make_archived_job(tmp_path)
    called = []

    monkeypatch.setattr(
        "rpbench.resume.resume_simulation",
        lambda _: called.append(True),
    )

    simulation_matrix.execute(
        [job], root.parent,
        dry_run=True,
        resume_failed=True,
    )

    assert not called
    state = json.loads((root / "job.json").read_text())
    assert state["status"] == "failed"
