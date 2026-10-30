"""Readable Markdown reports with separate lossless prompt/response archives."""

import argparse
import json
from pathlib import Path

from .refusal_experiment_report import (
    collect,
    fmt,
    md_code,
    summarize,
    summary_table,
)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def fenced_text(value):
    # Existing dynamic-fence renderer protects HTML/XML and embedded Markdown.
    return md_code(value)


def safe_name(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
        raise ValueError(f"Unsafe report filename: {value!r}")
    return value


def case_slug(sid):
    return safe_name(sid.removeprefix("refusal-pilot-v1-"))


def relative_link(label, target):
    return f"[{label}]({target})"


def render_judge(judgment, kind):
    record = judgment["record"]
    verdict = judgment["verdict"]

    lines = [
        f"#### {kind}",
        "",
        "| Feld | Bewertung |",
        "|---|---|",
    ]

    for key, value in verdict.items():
        if key in {"reason", "evidence"}:
            continue
        lines.append(f"| `{key}` | {fmt(value)} |")

    lines += [
        "",
        "**Begründung:**",
        "",
        fenced_text(verdict["reason"]),
    ]

    if "evidence" in verdict:
        lines += [
            "**Originalbeleg:**",
            "",
            fenced_text(verdict["evidence"]),
        ]

    if "repair" in record:
        lines += [
            "**Nachträgliche Belegkorrektur:**",
            "",
            fenced_text(
                json.dumps(record["repair"], ensure_ascii=False, indent=2)
            ),
        ]

    return lines


def render_scenario(data, sid, output):
    slug = case_slug(sid)
    scenario = data["scenarios"][sid]["data"]

    lines = [
        f"# {slug}",
        "",
        f"**Szenario-ID:** `{sid}`",
        "",
        f"**Kategorie:** `{scenario['category']}`",
        "",
        f"**Erwartetes Verhalten:** `{scenario['expected_behavior']}`",
        "",
        "[Zur Gesamtübersicht](../README.md)",
        "",
        "## Originale Gesprächshistorie",
        "",
        "### Systemanweisung",
        "",
        fenced_text(scenario["system"]),
    ]

    for i, message in enumerate(scenario["fixed_history"], 1):
        lines += [
            f"### Turn {i} — {message['role']}",
            "",
            fenced_text(message["content"]),
        ]

    lines += [
        "### Aktueller Player-Turn",
        "",
        fenced_text(scenario["turns"][2]),
        "",
        "## Direkter Vergleich",
        "",
        "| Modell | Preset | Refusal | Fulfillment | RP-Kontinuität | Output-Tokens |",
        "|---|---|---|---|---|---:|",
    ]

    for model in data["models"]:
        for preset in data["presets"]:
            item = data["results"][(model, preset)]["cases"][sid]
            refusal = item["judges"]["refusal"]["verdict"]
            fulfillment = item["judges"]["fulfillment"]["verdict"]
            tokens = (item["probe"].get("usage") or {}).get("completion_tokens")

            lines.append(
                f"| {model} | {preset} | "
                f"{fmt(refusal['model_refusal'])} | "
                f"{fulfillment['fulfillment']} | "
                f"{fmt(refusal['roleplay_continuity'])} | "
                f"{fmt(tokens)} |"
            )

    lines += ["", "## Antworten und Judge-Begründungen", ""]

    for model in data["models"]:
        for preset in data["presets"]:
            item = data["results"][(model, preset)]["cases"][sid]
            probe = item["probe"]

            stem = Path(safe_name(model)) / safe_name(preset) / slug
            prompt_path = output / "prompts" / stem.with_suffix(".md")
            raw_path = output / "raw-responses" / stem.with_suffix(".txt")
            meta_path = output / "metadata" / stem.with_suffix(".json")

            # Byte-for-byte UTF-8 text of the archived visible response.
            write(raw_path, probe["response"])

            prompt_messages = probe["messages"]
            original_messages = probe.get("preset_messages")

            prompt_lines = [
                f"# Effektiver Prompt — {model} / {preset} / {slug}",
                "",
                "## Tatsächlich gesendete Messages",
                "",
            ]

            for index, message in enumerate(prompt_messages, 1):
                prompt_lines += [
                    f"### {index}. {message['role']}",
                    "",
                    fenced_text(message["content"]),
                ]

            if original_messages:
                prompt_lines += [
                    "## Preset-Messages vor Backend-Kompatibilität",
                    "",
                ]
                for index, message in enumerate(original_messages, 1):
                    prompt_lines += [
                        f"### {index}. {message['role']}",
                        "",
                        fenced_text(message["content"]),
                    ]

            write(prompt_path, "\n".join(prompt_lines) + "\n")

            meta = {
                "scenario_id": sid,
                "model": model,
                "preset": preset,
                "probe_status": probe["status"],
                "finish_reason": probe.get("finish_reason"),
                "usage": probe.get("usage"),
                "reasoning_characters": len(probe.get("reasoning") or ""),
                "elapsed_seconds": probe.get("elapsed_seconds"),
                "probe_file": str(item["probe_path"]),
                "preset_metadata": probe.get("preset"),
                "compatibility_metadata": probe.get("preset_backend_compat"),
                "experiment_identity": data["results"][(model, preset)]["identity"],
                "judgments": {
                    kind: {
                        "source": str(judgment["path"]),
                        "invalid_previous_attempts": judgment["invalid_attempts"],
                        "repaired": "repair" in judgment["record"],
                    }
                    for kind, judgment in item["judges"].items()
                },
            }

            write(
                meta_path,
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            )

            prompt_link = f"../prompts/{stem.as_posix()}.md"
            raw_link = f"../raw-responses/{stem.as_posix()}.txt"
            meta_link = f"../metadata/{stem.as_posix()}.json"

            lines += [
                f"### {model} — {preset}",
                "",
                f"**Status:** `{probe['status']}` · "
                f"**Finish:** `{probe.get('finish_reason')}`",
                "",
                f"**Dateien:** {relative_link('Vollständiger Prompt', prompt_link)} · "
                f"{relative_link('Rohantwort', raw_link)} · "
                f"{relative_link('Technische Metadaten', meta_link)}",
                "",
                "#### Sichtbare Modellantwort",
                "",
                fenced_text(probe["response"]),
                "",
            ]

            lines += render_judge(item["judges"]["refusal"], "Refusal-Judge")
            lines += render_judge(item["judges"]["fulfillment"], "Fulfillment-Judge")
            lines += ["---", ""]

    write(output / "scenarios" / f"{slug}.md", "\n".join(lines) + "\n")


def export_v2(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    readme = [
        "# RPBench — 8K Benchmarkbericht",
        "",
        "## Datensatz",
        "",
        "- 2 Modelle, 7 Bedingungen, 16 Szenarien",
        "- 224 vollständige Generierungen",
        "- 448 gültige Judge-Urteile",
        "- Outputbudget: 8192 Tokens",
        f"- Experimentmatrix-SHA256: `{data['matrix_sha256']}`",
        "",
        "Fulfillment misst Anfrageerfüllung und nicht literarische Qualität. "
        "Inhaltsflags sind deskriptiv. Historische Presets können "
        "Backend-Kompatibilitätstransformationen enthalten.",
        "",
        "## Modell- und Preset-Vergleich",
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
        slug = case_slug(sid)
        readme.append(f"- [{slug}](scenarios/{slug}.md)")
        render_scenario(data, sid, output)

    readme += [
        "",
        "## Archivstruktur",
        "",
        "- `scenarios/`: vollständige Turns, Antworten und Judge-Begründungen",
        "- `prompts/`: effektive und ursprüngliche Preset-Messages",
        "- `raw-responses/`: unveränderte sichtbare Antworten",
        "- `metadata/`: Experimentidentitäten und technische Informationen",
        "",
    ]

    write(output / "README.md", "\n".join(readme) + "\n")

    print("Report:", output)
    print("Scenario reports:", len(data["scenario_order"]))
    print("Raw responses:", len(list((output / "raw-responses").rglob("*.txt"))))
    print("Prompt documents:", len(list((output / "prompts").rglob("*.md"))))
    print("Metadata documents:", len(list((output / "metadata").rglob("*.json"))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = collect(args.matrix, args.archive, args.judgments)
    export_v2(data, args.out)


if __name__ == "__main__":
    main()
