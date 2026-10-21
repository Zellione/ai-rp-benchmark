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

.player {background:#253044;border-left:4px solid #84a8df;
         padding:14px;margin:18px 0;white-space:pre-wrap}
.actions {display:flex;gap:9px;flex-wrap:wrap;margin:18px 0}
textarea {width:100%;box-sizing:border-box;min-height:85px}
.rating {display:grid;grid-template-columns:1fr 95px 95px;
         gap:12px;margin:9px 0;align-items:center}
small {color:#b1bdcf}
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
 <button id="export">Export JSON</button></div>
</header>
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
function render(){
 const item = DATA.turns[index];
 document.getElementById("progress").textContent =
   "Turn " + (index+1) + " / " + DATA.turns.length;
 const buttons = ["A","B","tie","skip"].map(value =>
   `<button data-vote="${value}" class="${
     votes[item.turn]===value?"selected":""}">${
     value==="tie"?"Tie":value==="skip"?"Skip":value
   }</button>`).join("");
 document.getElementById("turn").innerHTML = `
 <h2>Turn ${item.turn}</h2>
 <div class="player"><strong>Player</strong>
 <div class="rp-legend">Italic: action/narration · Bold: speech</div>
 <div class="player-text">${formatRP(item.player)}</div></div>
 <div class="cards">
  <div class="panel"><h2>A</h2><div class="answer">${
     formatRP(item.A)}</div></div>
  <div class="panel"><h2>B</h2><div class="answer">${
     formatRP(item.B)}</div></div>
 </div>
 <div class="actions"><strong>Which response do you prefer?</strong>
 ${buttons}</div>
 <textarea id="turnnote" placeholder="Optional reason...">${
     escapeHTML(notes[item.turn] || "")}</textarea>`;
 document.querySelectorAll("[data-vote]").forEach(button =>
  button.onclick = () => {
   votes[item.turn]=button.dataset.vote;
   render();
  });
 document.getElementById("turnnote").oninput = event =>
   notes[item.turn]=event.target.value;
 document.getElementById("prev").disabled = index===0;
 document.getElementById("next").disabled =
   index===DATA.turns.length-1;
}
function renderRatings(){
 const area = document.getElementById("ratings");
 area.innerHTML = DATA.criteria.map((name,i) => `
  <div class="rating"><label>${escapeHTML(name)}</label>
  ${["A","B"].map(side =>
   `<select data-key="${i}:${side}"><option value="">${side}: —</option>${
   [1,2,3,4,5].map(n => `<option value="${n}">${side}: ${n}</option>`
      ).join("")}</select>`).join("")}</div>`).join("");
 area.querySelectorAll("select").forEach(select =>
  select.onchange = () => ratings[select.dataset.key] =
    select.value==="" ? null : Number(select.value));
}
document.getElementById("prev").onclick=()=>{index--;render()};
document.getElementById("next").onclick=()=>{index++;render()};
document.getElementById("export").onclick=()=>{
 const result = {
  schema_version:1,
  comparison_id:DATA.id,
  mapping:DATA.mapping,
  method:DATA.method,
  turn_votes:DATA.turns.map(t=>({
    turn:t.turn,vote:votes[t.turn]||null,
    note:notes[t.turn]||""})),
  criteria:DATA.criteria.map((name,i)=>({
    name, A:ratings[i+":A"]??null,
    B:ratings[i+":B"]??null})),
  overall_preference:document.getElementById("overall").value||null,
  final_notes:document.getElementById("finalnotes").value
 };
 const blob=new Blob([JSON.stringify(result,null,2)],
   {type:"application/json"});
 const link=document.createElement("a");
 link.href=URL.createObjectURL(blob);
 link.download="rpbench-ab-"+DATA.id+".json";
 link.click();
 setTimeout(()=>URL.revokeObjectURL(link.href),2000);
};
renderRatings();
render();
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

    result = {
        "comparison_id": data["comparison_id"],
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
