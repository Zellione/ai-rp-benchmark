"""Auditable Markdown reporting for unified refusal experiments."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .refusal_experiment_judge import discover_experiments


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def md_code(value, language="text"):
    """Fence content without conflicts with embedded backticks."""
    value = str(value)
    runs = re.findall(r"`+", value)
    width = max([3, *(len(run) + 1 for run in runs)])
    fence = "`" * width
    return f"{fence}{language}\n{value.rstrip()}\n{fence}\n"


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Ja" if value else "Nein"
    return str(value)


def verdicts_for(probe_path, judgment_dir, kind):
    """Select exactly one valid verdict matching the probe bytes."""
    expected = sha256(probe_path)
    valid = []
    invalid_count = 0

    for path in sorted(Path(judgment_dir).glob("*.json")):
        row = read_json(path)

        if row.get("probe_sha256") != expected:
            raise ValueError(f"Judge/probe fingerprint mismatch: {path}")

        if row.get("valid") is True:
            valid.append((path, row))
        else:
            invalid_count += 1

    if len(valid) != 1:
        raise ValueError(
            f"{kind}: expected exactly one valid verdict, "
            f"found {len(valid)} in {judgment_dir}"
        )

    path, row = valid[0]
    if not isinstance(row.get("verdict"), dict):
        raise ValueError(f"Missing verdict: {path}")

    return {
        "path": path,
        "record": row,
        "verdict": row["verdict"],
        "invalid_attempts": invalid_count,
    }


def collect(matrix_path, archive, judgments):
    matrix = read_json(matrix_path)
    manifest_path = Path(matrix["scenario_manifest"])
    manifest = read_json(manifest_path)

    scenarios = {}
    order = []

    for item in manifest["cases"]:
        path = manifest_path.parent / item["file"]
        scenario = read_json(path)
        sid = scenario["id"]

        if sid != item["id"] or sid in scenarios:
            raise ValueError(f"Invalid scenario: {path}")

        scenarios[sid] = {
            "data": scenario,
            "sha256": sha256(path),
        }
        order.append(sid)

    experiments = discover_experiments(matrix_path, archive)
    results = {}
    model_order = [entry["id"] for entry in matrix["models"]]
    preset_order = [entry["id"] for entry in matrix["presets"]]

    for experiment in experiments:
        model = experiment["model"]
        preset = experiment["preset"]
        directory = experiment["directory"]

        if (model, preset) in results:
            raise ValueError(f"Duplicate experiment: {model}/{preset}")

        identity = read_json(directory / "experiment.json")

        by_scenario = {}
        for path in sorted(directory.glob("*/result.json")):
            probe = read_json(path)
            sid = probe["scenario_id"]

            if sid not in scenarios or sid in by_scenario:
                raise ValueError(f"Unexpected or repeated probe: {path}")

            if probe["scenario_sha256"] != scenarios[sid]["sha256"]:
                raise ValueError(f"Scenario digest mismatch: {path}")

            if probe.get("status") != "complete":
                raise ValueError(f"Incomplete generation: {path}")

            if probe.get("sampling", {}).get("max_tokens") != 8192:
                raise ValueError(f"Incorrect output budget: {path}")

            judgments_for_case = {}
            for kind in ("refusal", "fulfillment"):
                judge_dir = (
                    Path(judgments) / kind / directory.name / sid
                )
                judgments_for_case[kind] = verdicts_for(
                    path, judge_dir, kind
                )
                if (
                    judgments_for_case[kind]["record"].get("scenario_id")
                    != sid
                ):
                    raise ValueError(f"Judge scenario mismatch: {judge_dir}")

            by_scenario[sid] = {
                "probe": probe,
                "probe_path": path,
                "judges": judgments_for_case,
            }

        if set(by_scenario) != set(scenarios):
            raise ValueError(
                f"Incomplete experiment: {model}/{preset}, "
                f"{len(by_scenario)}/{len(scenarios)}"
            )

        results[(model, preset)] = {
            "identity": identity,
            "directory": directory,
            "cases": by_scenario,
        }

    expected_pairs = {
        (model, preset)
        for model in model_order
        for preset in preset_order
    }
    if set(results) != expected_pairs:
        raise ValueError("Missing or unexpected model/preset combinations")

    return {
        "matrix": matrix,
        "matrix_sha256": sha256(matrix_path),
        "scenarios": scenarios,
        "scenario_order": order,
        "models": model_order,
        "presets": preset_order,
        "results": results,
    }


def summarize(cases):
    refusal = Counter()
    fulfillment = Counter()
    completion_tokens = []
    reasoning_chars = []
    repairs = 0

    for item in cases.values():
        probe = item["probe"]
        jr = item["judges"]["refusal"]
        jf = item["judges"]["fulfillment"]
        rv = jr["verdict"]
        fv = jf["verdict"]

        refusal["refusals"] += int(rv["model_refusal"])
        refusal["continuity"] += int(rv["roleplay_continuity"])
        refusal["actionable"] += int(rv["unsafe_actionable"])
        refusal["sexual"] += int(rv["unsafe_sexual_content"])
        fulfillment[fv["fulfillment"]] += 1

        usage = probe.get("usage") or {}
        completion_tokens.append(usage.get("completion_tokens") or 0)
        reasoning_chars.append(len(probe.get("reasoning") or ""))
        repairs += int("repair" in jr["record"])
        repairs += int("repair" in jf["record"])

    return {
        **refusal,
        "fulfillment": dict(fulfillment),
        "avg_completion_tokens": round(
            sum(completion_tokens) / len(completion_tokens), 1
        ),
        "total_reasoning_characters": sum(reasoning_chars),
        "repaired_judgments": repairs,
    }


def summary_table(data, model):
    lines = [
        "| Preset | Refusals | Full | Substantial | Partial | None "
        "| RP-Kontinuität | Actionable | Sexual | Ø Output-Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for preset in data["presets"]:
        stats = summarize(data["results"][(model, preset)]["cases"])
        f = stats["fulfillment"]

        lines.append(
            f"| {preset} | {stats['refusals']}/16 "
            f"| {f.get('full', 0)} "
            f"| {f.get('substantial', 0)} "
            f"| {f.get('partial', 0)} "
            f"| {f.get('none', 0)} "
            f"| {stats['continuity']}/16 "
            f"| {stats['actionable']} "
            f"| {stats['sexual']} "
            f"| {stats['avg_completion_tokens']} |"
        )

    return "\n".join(lines)


def render_messages(messages):
    lines = []
    for index, message in enumerate(messages):
        lines += [
            f"#### Nachricht {index + 1}: {message['role']}",
            "",
            md_code(message["content"]),
            "",
        ]
    return lines


def render_case(data, sid):
    scenario = data["scenarios"][sid]["data"]
    slug = sid.removeprefix("refusal-pilot-v1-")
    lines = [
        f"# Szenario: {slug}",
        "",
        f"- ID: `{sid}`",
        f"- Kategorie: `{scenario['category']}`",
        f"- Erwartetes Verhalten: `{scenario['expected_behavior']}`",
        "",
        "## Originales Szenario und Gesprächsverlauf",
        "",
        "### System-Anweisungen",
        "",
        md_code(scenario["system"]),
        "",
    ]

    for i, turn in enumerate(scenario["fixed_history"], 1):
        lines += [
            f"### History-Turn {i}: {turn['role']}",
            "",
            md_code(turn["content"]),
            "",
        ]

    lines += [
        "### Abschließender Player-Turn",
        "",
        md_code(scenario["turns"][2]),
        "",
        "## Modellantworten und Bewertungen",
        "",
    ]

    for model in data["models"]:
        for preset in data["presets"]:
            item = data["results"][(model, preset)]["cases"][sid]
            probe = item["probe"]
            refusal = item["judges"]["refusal"]
            fulfillment = item["judges"]["fulfillment"]
            rv = refusal["verdict"]
            fv = fulfillment["verdict"]
            usage = probe.get("usage") or {}

            lines += [
                f"### {model} / {preset}",
                "",
                "| Eigenschaft | Wert |",
                "|---|---|",
                f"| Status | {fmt(probe['status'])} |",
                f"| Finish Reason | {fmt(probe.get('finish_reason'))} |",
                f"| Prompt-Tokens | {fmt(usage.get('prompt_tokens'))} |",
                f"| Completion-Tokens | {fmt(usage.get('completion_tokens'))} |",
                f"| Reasoning-Zeichen | {len(probe.get('reasoning') or '')} |",
                f"| Dauer (s) | {fmt(probe.get('elapsed_seconds'))} |",
                f"| Refusal | {fmt(rv['model_refusal'])} |",
                f"| Refusal-Stil | {fmt(rv['refusal_style'])} |",
                f"| RP-Kontinuität | {fmt(rv['roleplay_continuity'])} |",
                f"| Actionable-Flag | {fmt(rv['unsafe_actionable'])} |",
                f"| Sexual-Content-Flag | {fmt(rv['unsafe_sexual_content'])} |",
                f"| Fulfillment | {fmt(fv['fulfillment'])} |",
                f"| Antwortmodus | {fmt(fv['response_mode'])} |",
                "",
                "**Refusal-Judge – Begründung:**",
                "",
                rv["reason"],
                "",
                "**Fulfillment-Judge – Begründung:**",
                "",
                fv["reason"],
                "",
                "**Refusal-Evidence:**",
                "",
                md_code(rv["evidence"]),
                "",
            ]

            for kind, judgment in (
                ("Refusal", refusal),
                ("Fulfillment", fulfillment),
            ):
                if "repair" in judgment["record"]:
                    lines += [
                        f"**{kind}-Urteil nachträglich korrigiert:**",
                        "",
                        md_code(
                            json.dumps(
                                judgment["record"]["repair"],
                                ensure_ascii=False, indent=2
                            ),
                            "json",
                        ),
                        "",
                    ]

            lines += [
                "#### Vollständige Modellantwort",
                "",
                md_code(probe["response"]),
                "",
                '#### Effektiv gesendeter Modellprompt',
                "",
                *render_messages(probe["messages"]),
                "",
            ]

            original = probe.get("preset_messages")
            if original:
                lines += [
                    '#### Originale Preset-Messages vor Backend-Lowering',
                    "",
                    *render_messages(original),
                    "",
                ]

            lines += [
                '#### Experiment- und Preset-Metadaten',
                "",
                md_code(json.dumps({
                    "experiment": data["results"][(model, preset)]["identity"],
                    "preset": probe.get("preset"),
                    "preset_backend_compat": probe.get(
                        "preset_backend_compat"
                    ),
                    "probe_sha256": sha256(item["probe_path"]),
                }, indent=2, ensure_ascii=False), "json"),
                "",
                "",
                "---",
                "",
            ]

    return "\n".join(lines)


def export(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scenarios").mkdir(exist_ok=True)

    total_repairs = sum(
        summarize(experiment["cases"])["repaired_judgments"]
        for experiment in data["results"].values()
    )

    readme = [
        "# RPBench – Refusal-Pilot 8K",
        "",
        "## Experimentumfang",
        "",
        f"- Modelle: {len(data['models'])}",
        f"- Preset-Bedingungen: {len(data['presets'])}",
        f"- Szenarien: {len(data['scenarios'])}",
        "- Outputbudget: 8.192 Tokens",
        "- Vollständige Generierungen: 224/224",
        "- Gültige Refusal- und Fulfillment-Urteile: 448/448",
        f"- Nachträglich korrigierte Evidence-Belege: {total_repairs}",
        f"- Experimentmatrix SHA-256: `{data['matrix_sha256']}`",
        "",
        "## Methodik",
        "",
        "Jede Modell-/Preset-Kombination wird mit denselben 16 "
        "Fixed-History-Szenarien ausgewertet. Fulfillment misst "
        "Anfrageerfüllung, nicht literarische Qualität. Die Inhaltsflags "
        "sind deskriptiv und keine Qualitätsbewertung. "
        "Preset-Adapter können von der exakten SillyTavern-Serialisierung "
        "abweichen. Details stehen in den Experimentmetadaten.",
        "",
        "## Vergleich nach Modell",
        "",
    ]

    for model in data["models"]:
        readme += [
            f"### {model}",
            "",
            summary_table(data, model),
            "",
        ]

    readme += ["## Szenarioberichte", ""]
    for sid in data["scenario_order"]:
        slug = sid.removeprefix("refusal-pilot-v1-")
        readme.append(
            f"- [{slug}](scenarios/{slug}.md)"
        )

    (output / "README.md").write_text(
        "\n".join(readme) + "\n",
        encoding="utf-8",
    )

    for model in data["models"]:
        content = "\n".join([
            f"# {model} – 8K Preset-Vergleich",
            "",
            summary_table(data, model),
            "",
            "Detailantworten und vollständige Prompts stehen "
            "in den Szenarioberichten.",
            "",
        ])
        (output / f"{model}.md").write_text(
            content, encoding="utf-8"
        )

    for sid in data["scenario_order"]:
        slug = sid.removeprefix("refusal-pilot-v1-")
        (output / "scenarios" / f"{slug}.md").write_text(
            render_case(data, sid),
            encoding="utf-8",
        )

    print("Report:", output)
    print("Markdown files:", len(list(output.rglob("*.md"))))
    print("Repaired judgments:", total_repairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = collect(
        args.matrix,
        args.archive,
        args.judgments,
    )
    export(data, args.out)


if __name__ == "__main__":
    main()
