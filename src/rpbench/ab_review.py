"""Local, blind, human A/B review of archived RP conversations."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from .observations import read_observations, source_path


CRITERIA = [
    "Character fidelity",
    "Writing quality",
    "Immersion",
    "Initiative",
    "Emotional credibility",
]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_case_context(run_dir):
    """Read the exact prompts used by a controlled NPC run."""
    root = Path(run_dir)
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )

    if manifest.get("type") != "fixed_context_npc_ab":
        return None

    count = manifest.get("completed_turns")
    if type(count) is not int or count < 1:
        raise ValueError("Incomplete fixed-context NPC archive")

    contexts = []
    for index in range(1, count + 1):
        record = json.loads(
            (
                root / "cases" / f"{index:03d}.json"
            ).read_text(encoding="utf-8")
        )
        payload = record.get("payload", {})
        messages = payload.get("messages")

        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError("Invalid fixed-case messages")

        for position, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError("Invalid fixed-case message")
            if (
                message.get("role") not in {
                    "system", "user", "assistant"
                }
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("Invalid fixed-case message")

            if position == 0 and message["role"] != "system":
                raise ValueError("Missing system prompt")

        if messages[-1]["role"] != "user":
            raise ValueError("Missing final player action")

        contexts.append({
            "id": record.get("id"),
            "messages": messages,
        })

    return contexts


def prepare(run_a, run_b, seed="personal-rp"):
    a = Path(run_a).resolve()
    b = Path(run_b).resolve()

    if a == b:
        raise ValueError("Choose two different runs")

    left = read_observations(a)
    right = read_observations(b)

    if not left or not right:
        raise ValueError("Runs must contain completed turns")

    if len(left) != len(right):
        raise ValueError("Runs contain different turn counts")

    for x, y in zip(left, right):
        if (x["turn"], x["role"]) != (y["turn"], y["role"]):
            raise ValueError("Runs have incompatible turn sequences")
        if x["role"] == "player" and x["text"] != y["text"]:
            raise ValueError(
                f"Player input differs at turn {x['turn']}. "
                "Turnwise A/B requires identical player inputs."
            )

    fixed_a = fixed_case_context(a)
    fixed_b = fixed_case_context(b)

    if (fixed_a is None) != (fixed_b is None):
        raise ValueError("Cannot mix fixed-context and self-play runs")

    if fixed_a is not None:
        if len(fixed_a) != len(left) // 2:
            raise ValueError("Fixed-context case count mismatch")

        if fixed_a != fixed_b:
            raise ValueError(
                "Controlled A/B requires identical archived inputs"
            )

        for index, case in enumerate(fixed_a):
            if (
                case["messages"][-1]["content"]
                != left[2 * index]["text"]
            ):
                raise ValueError(
                    "Archived player action differs from request"
                )

    identity = hashlib.sha256(
        (seed + "\0" + str(a) + "\0" + str(b)).encode()
    ).digest()

    if identity[0] % 2:
        a, b = b, a
        left, right = right, left

    turns = []
    for i in range(0, len(left), 2):
        turns.append({
            "turn": left[i]["turn"],
            "player": left[i]["text"],
            "context": (
                fixed_a[i // 2]["messages"][1:-1]
                if fixed_a is not None
                else None
            ),
            "A": left[i + 1]["text"],
            "B": right[i + 1]["text"],
        })

    return {
        "schema_version": 1,
        "id": hashlib.sha256(
            json.dumps(
                {
                    "runs": [str(a), str(b)],
                    "source_hashes": [
                        file_hash(source_path(a)),
                        file_hash(source_path(b)),
                    ],
                    "seed": seed,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16],
        "turns": turns,
        "criteria": CRITERIA,
        "mapping": {
            "A": {
                "run": str(a),
                "source_sha256": file_hash(source_path(a)),
            },
            "B": {
                "run": str(b),
                "source_sha256": file_hash(source_path(b)),
            },
        },
        "method": (
            "Personal blind A/B; identical player prompts; "
            "human preference, not an objective model score"
        ),
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RPBench — Blind A/B Review</title>
<style>
:root {color-scheme:dark;font-family:system-ui,sans-serif}
body {background:#11151c;color:#e8edf5;margin:0 auto;
      padding:24px;max-width:1350px}
header {display:flex;justify-content:space-between;
        align-items:center;gap:15px;flex-wrap:wrap}
button,select,textarea {background:#222b38;color:#e8edf5;
   border:1px solid #536174;border-radius:7px;padding:9px;
   font:inherit}
button {cursor:pointer}
button.selected {background:#225b9a;border-color:#79b8ff}
button:hover {border-color:#a8c8e8}
.cards {display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel {background:#1a202a;padding:18px;border-radius:12px;
        border:1px solid #394657;min-width:0}
.answer,.player-text {white-space:pre-wrap;line-height:1.7;
         overflow-wrap:anywhere}
.rp-dialogue {color:#f1f5fc;font-weight:600}
.rp-action {color:#b7c9df;font-style:italic}
.rp-plain {color:#e0e6f0}
.rp-legend {margin:8px 0;color:#aebcd0;font-size:.85rem}

.case-context {
 background:#192635;border:1px solid #40556a;
 border-radius:10px;padding:12px 16px;margin:16px 0;
}
.case-context summary {cursor:pointer;font-weight:600}
.context-entry {margin:14px 0;line-height:1.6}
.context-entry pre {
 white-space:pre-wrap;overflow-wrap:anywhere;
 font:inherit;color:#e0e6f0;margin:6px 0;
}
.player {background:#253044;border-left:4px solid #84a8df;
         padding:14px;margin:18px 0;white-space:pre-wrap}
.actions {display:flex;gap:9px;flex-wrap:wrap;margin:18px 0}
textarea {width:100%;box-sizing:border-box;min-height:85px}
.rating {display:grid;grid-template-columns:1fr 95px 95px;
         gap:12px;margin:9px 0;align-items:center}
small {color:#b1bdcf}
.review-toolbar {display:flex;align-items:center;gap:12px;
 flex-wrap:wrap;margin:18px 0}
.review-toolbar label {display:flex;align-items:center;gap:8px}
.review-toolbar input {max-width:95px;background:#222b38;
 color:#e8edf5;border:1px solid #536174;border-radius:7px;
 padding:9px;font:inherit}
#review-progress {width:100%;height:12px;accent-color:#4386be}
#storage-status {color:#b1bdcf;font-size:.85rem}
@media(max-width:780px){
 .cards {grid-template-columns:1fr}
}
</style>
</head>
<body>
<header>
 <div><h1>RPBench · Blind A/B Review</h1>
 <small id="progress"></small></div>
 <div><button id="prev">← Previous</button>
 <button id="next">Next →</button>
 <button id="open-turn">Next unrated</button>
 <button id="export">Export JSON</button>
 <button id="import-button">Import JSON</button></div>
</header>
<div class="review-toolbar">
 <label>Jump to turn
 <input id="jump" type="number" min="1" step="1"></label>
 <button id="jump-button">Go</button>
 <label>Review status
 <select id="review-status">
 <option value="in_progress">In progress</option>
 <option value="complete">Complete</option>
 </select></label>
 <span id="storage-status" role="status"></span>
</div>
<progress id="review-progress" value="0" max="1"></progress>
<input id="import-file" type="file" accept=".json,application/json"
 hidden>
<section id="turn"></section>
<h2>Overall impressions</h2>
<p><small>Rate the complete conversations from 1 (poor)
to 5 (excellent). Optional.</small></p>
<div id="ratings"></div>
<p>Overall preference:
<select id="overall"><option value="">Not rated</option>
<option value="A">A</option><option value="B">B</option>
<option value="tie">Tie</option></select></p>
<p>Final notes:</p>
<textarea id="finalnotes" placeholder="What felt better and why?"></textarea>
<p><small>Responses stay in this browser session until exported.
Export before closing the page. Model identities are hidden
during review; the exported JSON contains the run mapping.</small></p>
<script>
const DATA = __RPBENCH_DATA__;
let index = 0;
const votes = {};
const notes = {};
const ratings = {};
const escapeHTML = value => {
 const el = document.createElement("div");
 el.textContent = value;
 return el.innerHTML;
};
function formatRP(text) {
 // Presentation only: never change or infer the original content.
 // Recognizes *actions* and quoted speech, including curly quotes.
 const expression = /(\*[^*\n]+\*|“[^”\n]+”|"[^"\n]+")/g;
 const parts = [];
 let cursor = 0;

 for (const match of text.matchAll(expression)) {
  const start = match.index;
  if (start > cursor) {
   parts.push('<span class="rp-plain">' +
     escapeHTML(text.slice(cursor, start)) + '</span>');
  }
  const value = match[0];
  if (value.startsWith("*")) {
   parts.push('<span class="rp-action">' +
     escapeHTML(value.slice(1, -1)) + '</span>');
  } else {
   parts.push('<span class="rp-dialogue">' +
     escapeHTML(value) + '</span>');
  }
  cursor = start + value.length;
 }
 if (cursor < text.length) {
  parts.push('<span class="rp-plain">' +
    escapeHTML(text.slice(cursor)) + '</span>');
 }
 return parts.join("");
}
const STORAGE_KEY = "rpbench-ab-v2:" + DATA.id;
let reviewStatus = "in_progress";
let storageAvailable = false;

function snapshot() {
 return {
  schema_version: 2,
  comparison_id: DATA.id,
  mapping: DATA.mapping,
  method: DATA.method,
  status: reviewStatus,
  current_turn: DATA.turns[index].turn,
  turn_votes: DATA.turns.map(t => ({
   turn: t.turn,
   vote: votes[t.turn] || null,
   note: notes[t.turn] || ""
  })),
  criteria: DATA.criteria.map((name, i) => ({
   name,
   A: ratings[i + ":A"] ?? null,
   B: ratings[i + ":B"] ?? null
  })),
  overall_preference:
   document.getElementById("overall").value || null,
  final_notes:
   document.getElementById("finalnotes").value
 };
}

function save() {
 const status = document.getElementById("storage-status");
 try {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot()));
  storageAvailable = true;
  status.textContent = "Autosaved locally";
 } catch (err) {
  storageAvailable = false;
  status.textContent =
   "Browser autosave unavailable — export JSON regularly";
 }
 updateProgress();
}

function restore(data) {
 if (!data || data.comparison_id !== DATA.id) {
  throw new Error("Comparison ID does not match this review");
 }
 if (JSON.stringify(data.mapping) !==
     JSON.stringify(DATA.mapping)) {
  throw new Error("Run mapping / source hashes do not match");
 }
 if (!Array.isArray(data.turn_votes) ||
     !Array.isArray(data.criteria)) {
  throw new Error("Invalid review data");
 }
 const permitted = new Set(["A","B","tie","skip",null]);
 const expected = new Set(DATA.turns.map(t => t.turn));
 const restoredVotes = {};
 const restoredNotes = {};
 const restoredRatings = {};

 if (data.turn_votes.length !== DATA.turns.length) {
  throw new Error("Turn count mismatch");
 }
 for (const item of data.turn_votes) {
  if (!expected.has(item.turn) ||
      !permitted.has(item.vote ?? null) ||
      Object.hasOwn(restoredVotes, item.turn)) {
   throw new Error("Invalid or duplicate turn vote");
  }
  restoredVotes[item.turn] = item.vote ?? null;
  restoredNotes[item.turn] =
   typeof item.note === "string" ? item.note : "";
 }
 for (let i = 0; i < DATA.criteria.length; i++) {
  const entry = data.criteria[i];
  if (!entry || entry.name !== DATA.criteria[i]) {
   throw new Error("Review criteria mismatch");
  }
  for (const side of ["A","B"]) {
   const value = entry[side];
   if (value != null &&
       (!Number.isInteger(value) || value < 1 || value > 5)) {
    throw new Error("Invalid criterion rating");
   }
   restoredRatings[i + ":" + side] = value ?? null;
  }
 }
 const preference = data.overall_preference ?? "";
 if (!["","A","B","tie"].includes(preference)) {
  throw new Error("Invalid overall preference");
 }
 const nextStatus = data.status ?? "in_progress";
 if (!["in_progress","complete"].includes(nextStatus)) {
  throw new Error("Invalid review status");
 }

 // Apply only after all validation has succeeded.
 for (const key of Object.keys(votes)) delete votes[key];
 for (const key of Object.keys(notes)) delete notes[key];
 for (const key of Object.keys(ratings)) delete ratings[key];
 Object.assign(votes, restoredVotes);
 Object.assign(notes, restoredNotes);
 Object.assign(ratings, restoredRatings);
 reviewStatus = nextStatus;
 const position = DATA.turns.findIndex(
  t => t.turn === data.current_turn
 );
 index = position >= 0 ? position : 0;
 document.getElementById("overall").value = preference;
 document.getElementById("finalnotes").value =
  typeof data.final_notes === "string" ? data.final_notes : "";
 document.getElementById("review-status").value = reviewStatus;
 renderRatings();
 render();
 save();
}

function updateProgress() {
 const reviewed = DATA.turns.filter(
  t => ["A","B","tie"].includes(votes[t.turn])
 ).length;
 const skipped = DATA.turns.filter(
  t => votes[t.turn] === "skip"
 ).length;
 const progress = document.getElementById("review-progress");
 progress.max = DATA.turns.length;
 progress.value = reviewed;
 document.getElementById("progress").textContent =
  `Turn ${index+1}/${DATA.turns.length} · ` +
  `${reviewed} rated · ${skipped} skipped · ` +
  `${DATA.turns.length-reviewed-skipped} open`;
 document.getElementById("jump").value =
  DATA.turns[index].turn;
}

function render() {
 const item = DATA.turns[index];
 const buttons = ["A","B","tie","skip"].map(value =>
  `<button data-vote="${value}" class="${
   votes[item.turn] === value ? "selected" : ""}">${
   value === "tie" ? "Tie" :
   value === "skip" ? "Skip" : value
  }</button>`
 ).join("");
 document.getElementById("turn").innerHTML = `
 <h2>Turn ${item.turn}</h2>
 ${item.context ? `
 <details class="case-context">
 <summary>Shared scene context / Vorgeschichte</summary>
 ${item.context.map(message => `
 <div class="context-entry">
 <strong>${message.role === "user" ? "Player / Context" : "NPC"}</strong>
 <pre>${escapeHTML(message.content)}</pre>
 </div>`).join("")}
 </details>` : ""}
 <div class="player"><strong>Player</strong>
 <div class="rp-legend">Italic: action/narration · Bold: speech</div>
 <div class="player-text">${formatRP(item.player)}</div></div>
 <div class="cards">
 <div class="panel"><h2>A</h2><div class="answer">${
  formatRP(item.A)}</div></div>
 <div class="panel"><h2>B</h2><div class="answer">${
  formatRP(item.B)}</div></div>
 </div>
 <div class="actions"><strong>Preferred response:</strong>
 ${buttons}</div>
 <textarea id="turnnote" placeholder="Optional reason...">${
  escapeHTML(notes[item.turn] || "")}</textarea>`;

 document.querySelectorAll("[data-vote]").forEach(button => {
  button.onclick = () => {
   votes[item.turn] = button.dataset.vote;
   render();
   save();
  };
 });
 document.getElementById("turnnote").oninput = event => {
  notes[item.turn] = event.target.value;
  save();
 };
 document.getElementById("prev").disabled = index === 0;
 document.getElementById("next").disabled =
  index === DATA.turns.length - 1;
 updateProgress();
}

function renderRatings() {
 const area = document.getElementById("ratings");
 area.innerHTML = DATA.criteria.map((name, i) => `
 <div class="rating"><label>${escapeHTML(name)}</label>
 ${["A","B"].map(side =>
  `<select data-key="${i}:${side}">
   <option value="">${side}: —</option>
   ${[1,2,3,4,5].map(n =>
    `<option value="${n}">${side}: ${n}</option>`
   ).join("")}</select>`
 ).join("")}</div>`).join("");
 area.querySelectorAll("select").forEach(select => {
  const value = ratings[select.dataset.key];
  select.value = value == null ? "" : String(value);
  select.onchange = () => {
   ratings[select.dataset.key] =
    select.value === "" ? null : Number(select.value);
   save();
  };
 });
}

function goTo(position) {
 if (position < 0 || position >= DATA.turns.length) return;
 index = position;
 render();
 save();
}

document.getElementById("prev").onclick = () =>
 goTo(index - 1);
document.getElementById("next").onclick = () =>
 goTo(index + 1);

document.getElementById("open-turn").onclick = () => {
 const next = DATA.turns.findIndex(
  (t, i) => i > index && !votes[t.turn]
 );
 const fallback = DATA.turns.findIndex(t => !votes[t.turn]);
 if (next >= 0) goTo(next);
 else if (fallback >= 0) goTo(fallback);
 else alert("No unrated turns remain.");
};

document.getElementById("jump-button").onclick = () => {
 const turn = Number(document.getElementById("jump").value);
 const position = DATA.turns.findIndex(t => t.turn === turn);
 if (position < 0) alert("Turn not found.");
 else goTo(position);
};

document.getElementById("jump").onkeydown = event => {
 if (event.key === "Enter")
  document.getElementById("jump-button").click();
};

document.getElementById("review-status").onchange = event => {
 reviewStatus = event.target.value;
 save();
};
document.getElementById("overall").onchange = save;
document.getElementById("finalnotes").oninput = save;

document.getElementById("export").onclick = () => {
 const blob = new Blob(
  [JSON.stringify(snapshot(), null, 2)],
  {type:"application/json"}
 );
 const link = document.createElement("a");
 link.href = URL.createObjectURL(blob);
 link.download = "rpbench-ab-" + DATA.id + ".json";
 link.click();
 setTimeout(() => URL.revokeObjectURL(link.href), 2000);
};

document.getElementById("import-button").onclick = () =>
 document.getElementById("import-file").click();

document.getElementById("import-file").onchange = async event => {
 const file = event.target.files[0];
 if (!file) return;
 try {
  const contents = JSON.parse(await file.text());
  restore(contents);
  alert("Review restored successfully.");
 } catch (err) {
  alert("Import failed: " + err.message);
 }
 event.target.value = "";
};

renderRatings();
render();

try {
 const saved = localStorage.getItem(STORAGE_KEY);
 if (saved) restore(JSON.parse(saved));
 else save();
} catch (err) {
 document.getElementById("storage-status").textContent =
  "Autosave unavailable — use JSON export/import";
}
</script>
</body>
</html>"""


def export_html(data, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False).replace(
        "<", "\\u003c"
    ).replace(">", "\\u003e").replace("&", "\\u0026")
    output.write_text(
        PAGE.replace("__RPBENCH_DATA__", encoded),
        encoding="utf-8",
    )


def summarize(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    votes = {"A": 0, "B": 0, "tie": 0, "skip": 0}
    for row in data["turn_votes"]:
        vote = row.get("vote")
        if vote in votes:
            votes[vote] += 1

    rows = data["turn_votes"]
    rated = votes["A"] + votes["B"] + votes["tie"]
    result = {
        "comparison_id": data["comparison_id"],
        "status": data.get("status", "complete"),
        "rated_turns": rated,
        "total_turns": len(rows),
        "completion_fraction": (
            rated / len(rows) if rows else 0.0
        ),
        "votes": votes,
        "overall_preference": data.get("overall_preference"),
        "criteria": data.get("criteria", []),
        "mapping": data["mapping"],
        "final_notes": data.get("final_notes", ""),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("create")
    make.add_argument("--run-a", required=True)
    make.add_argument("--run-b", required=True)
    make.add_argument("--out", required=True)
    make.add_argument("--seed", default="personal-rp")

    report = sub.add_parser("report")
    report.add_argument("--results", required=True)

    args = parser.parse_args()
    if args.command == "create":
        data = prepare(args.run_a, args.run_b, args.seed)
        export_html(data, args.out)
        print("Created:", args.out)
        print("Turns:", len(data["turns"]))
    else:
        print(json.dumps(
            summarize(args.results),
            indent=2, ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
