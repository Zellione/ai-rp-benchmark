"""Blind, resumable review of independently evolving RP conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .observations import read_observations, source_path


CRITERIA = [
    "Character fidelity",
    "Dialogue quality",
    "Narration quality",
    "Initiative",
    "Story consistency",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conversation(run):
    entries = read_observations(run)
    if not entries or len(entries) % 2:
        raise ValueError("Run has no complete conversation")
    return [
        {
            "turn": entries[i]["turn"],
            "player": entries[i]["text"],
            "npc": entries[i + 1]["text"],
        }
        for i in range(0, len(entries), 2)
    ]


def prepare(run_a, run_b, section_size=20, seed="personal-rp"):
    if type(section_size) is not int or not 1 <= section_size <= 100:
        raise ValueError("section_size must be between 1 and 100")

    a, b = Path(run_a).resolve(), Path(run_b).resolve()
    if a == b:
        raise ValueError("Choose two different runs")

    ca, cb = conversation(a), conversation(b)
    fingerprint = [
        {"run": str(p), "source_sha256": sha(source_path(p))}
        for p in (a, b)
    ]

    # Deterministic blinding, without requiring matching player prompts.
    decision = hashlib.sha256(
        (seed + "\0" + str(a) + "\0" + str(b)).encode()
    ).digest()[0] % 2

    if decision:
        ca, cb = cb, ca
        fingerprint.reverse()

    count = max(
        (len(ca) + section_size - 1) // section_size,
        (len(cb) + section_size - 1) // section_size,
    )

    sections = [
        {
            "id": i + 1,
            "A": ca[i * section_size:(i + 1) * section_size],
            "B": cb[i * section_size:(i + 1) * section_size],
        }
        for i in range(count)
    ]

    identity = {
        "mapping": {"A": fingerprint[0], "B": fingerprint[1]},
        "section_size": section_size,
        "seed": seed,
    }
    comparison_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:16]

    return {
        "schema_version": 1,
        "id": comparison_id,
        "section_size": section_size,
        "turn_counts": {"A": len(ca), "B": len(cb)},
        "sections": sections,
        "criteria": CRITERIA,
        "mapping": identity["mapping"],
        "method": (
            "Blind independent-conversation review; sections reflect "
            "relative progress, not equivalent story events"
        ),
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RPBench · Conversation Review</title>
<style>
:root {color-scheme:dark;font-family:system-ui,sans-serif}
body {background:#11151c;color:#e7edf6;max-width:1450px;
 margin:auto;padding:22px}
header,.toolbar {display:flex;align-items:center;justify-content:
 space-between;flex-wrap:wrap;gap:12px;margin-bottom:18px}
button,select,input,textarea {font:inherit;color:inherit;
 background:#232d3c;border:1px solid #50617a;border-radius:7px;
 padding:9px}
button {cursor:pointer}
button:hover {border-color:#93b5da}
button.selected {background:#265e9b;border-color:#88baf1}
.cards {display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel {background:#1b2430;padding:16px;border-radius:12px;
 min-width:0;border:1px solid #36455a}
.entry {border-top:1px solid #3b495b;padding:12px 0}
.player {background:#253449;padding:12px;border-radius:8px;
 margin:8px 0}
.npc {padding:12px}
.rp-text {white-space:pre-wrap;overflow-wrap:anywhere;
 line-height:1.7}
.rp-action {font-style:italic;color:#b8cbe4}
.rp-dialogue {font-weight:650;color:#fff}
.rp-plain {color:#e0e6ef}
.controls {margin:16px 0}
.controls button {margin:4px}
textarea {width:100%;box-sizing:border-box;min-height:85px}
.rating {display:grid;grid-template-columns:1fr 90px 90px;
 gap:10px;align-items:center;margin:8px 0}
small,.muted {color:#b2c0d2}
progress {width:100%;height:13px;accent-color:#4b93d9}
#storage {font-size:.85rem}
@media(max-width:850px){.cards{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
 <div><h1>RPBench · Conversation Review</h1>
 <div id="progress-label" class="muted"></div></div>
 <div>
 <button id="prev">← Previous</button>
 <button id="next">Next →</button>
 <button id="unrated">Next unrated</button>
 <button id="export">Export JSON</button>
 <button id="import">Import JSON</button>
 </div>
</header>
<div class="toolbar">
 <label>Section <select id="jump"></select></label>
 <label>Status <select id="status">
 <option value="in_progress">In progress</option>
 <option value="complete">Complete</option>
 </select></label>
 <span id="storage"></span>
</div>
<progress id="progress" value="0" max="1"></progress>
<p class="muted">A and B are independent conversations. Section
numbers represent relative progress, not matching story events.
<em>*Actions*</em> are italic; <strong>"speech"</strong> is bold.</p>
<input id="file" type="file" accept=".json,application/json" hidden>
<main id="section"></main>
<h2>Overall impression</h2>
<p class="muted">Optional ratings for the complete conversation
(1 = poor, 5 = excellent).</p>
<div id="criteria"></div>
<p>Overall preference:
 <select id="overall">
  <option value="">Not rated</option>
  <option value="A">A</option>
  <option value="B">B</option>
  <option value="tie">Tie</option>
 </select>
</p>
<textarea id="final-note" placeholder="Overall strengths and weaknesses"></textarea>
<p class="muted">Autosave is browser-local. Export JSON for a durable
backup, especially when reviewing over several days.</p>
<script>
const DATA = __RPBENCH_DATA__;
const KEY = "rpbench-conversation-v1:" + DATA.id;
let index = 0;
let status = "in_progress";
const votes = {};
const notes = {};
const ratings = {};

const el = id => document.getElementById(id);
function escapeHTML(text) {
 const node = document.createElement("div");
 node.textContent = text;
 return node.innerHTML;
}
function formatRP(text) {
 const pattern = /(\*[^*\n]+\*|“[^”\n]+”|"[^"\n]+")/g;
 let html = "", cursor = 0;
 for (const match of text.matchAll(pattern)) {
  html += '<span class="rp-plain">' +
   escapeHTML(text.slice(cursor,match.index)) + '</span>';
  const value = match[0];
  html += value.startsWith("*")
   ? '<span class="rp-action">' +
     escapeHTML(value.slice(1,-1)) + '</span>'
   : '<span class="rp-dialogue">' +
     escapeHTML(value) + '</span>';
  cursor = match.index + value.length;
 }
 return html + '<span class="rp-plain">' +
  escapeHTML(text.slice(cursor)) + '</span>';
}
function showConversation(side, turns) {
 if (!turns.length) return '<p class="muted">No turns in this section.</p>';
 return turns.map(t => `
 <article class="entry">
 <h4>Turn ${t.turn}</h4>
 <div class="player"><small>Player</small>
 <div class="rp-text">${formatRP(t.player)}</div></div>
 <div class="npc"><small>NPC</small>
 <div class="rp-text">${formatRP(t.npc)}</div></div>
 </article>`).join("");
}
function progress() {
 const rated = DATA.sections.filter(
  s => ["A","B","tie"].includes(votes[s.id])
 ).length;
 const skipped = DATA.sections.filter(
  s => votes[s.id] === "skip"
 ).length;
 el("progress").max = DATA.sections.length;
 el("progress").value = rated;
 el("progress-label").textContent =
  `Section ${index+1}/${DATA.sections.length} · ` +
  `${rated} rated · ${skipped} skipped · ` +
  `${DATA.sections.length-rated-skipped} open`;
 el("jump").value = String(DATA.sections[index].id);
}
function snapshot() {
 return {
  schema_version: 1,
  comparison_id: DATA.id,
  mapping: DATA.mapping,
  section_size: DATA.section_size,
  turn_counts: DATA.turn_counts,
  status: status,
  current_section: DATA.sections[index].id,
  sections: DATA.sections.map(s => ({
   id: s.id,
   preference: votes[s.id] ?? null,
   note: notes[s.id] ?? ""
  })),
  criteria: DATA.criteria.map((name,i) => ({
   name,
   A: ratings[i+":A"] ?? null,
   B: ratings[i+":B"] ?? null
  })),
  overall_preference: el("overall").value || null,
  final_notes: el("final-note").value
 };
}
function save() {
 try {
  localStorage.setItem(KEY, JSON.stringify(snapshot()));
  el("storage").textContent = "Saved locally";
 } catch (_) {
  el("storage").textContent =
   "Autosave unavailable: export JSON regularly";
 }
 progress();
}
function render() {
 const section = DATA.sections[index];
 el("section").innerHTML = `
  <h2>Section ${section.id}</h2>
  <div class="cards">
   <div class="panel"><h2>A</h2>${
    showConversation("A",section.A)}</div>
   <div class="panel"><h2>B</h2>${
    showConversation("B",section.B)}</div>
  </div>
  <div class="controls">Which section do you prefer?
   ${["A","B","tie","skip"].map(v =>
    `<button data-vote="${v}" class="${
     votes[section.id]===v?"selected":""}">${
      v==="tie"?"Tie":v==="skip"?"Skip":v}</button>`
   ).join("")}
  </div>
  <textarea id="section-note" placeholder="Optional section notes">${
   escapeHTML(notes[section.id] ?? "")}</textarea>`;
 document.querySelectorAll("[data-vote]").forEach(b => {
  b.onclick = () => {
   votes[section.id] = b.dataset.vote;
   render();
   save();
  };
 });
 el("section-note").oninput = e => {
  notes[section.id] = e.target.value;
  save();
 };
 el("prev").disabled = index === 0;
 el("next").disabled = index === DATA.sections.length-1;
 progress();
}
function renderRatings() {
 el("criteria").innerHTML = DATA.criteria.map((name,i) =>
  `<div class="rating"><label>${escapeHTML(name)}</label>${
   ["A","B"].map(side =>
    `<select data-rating="${i}:${side}">
    <option value="">${side}: —</option>${
     [1,2,3,4,5].map(n =>
      `<option value="${n}">${side}: ${n}</option>`
     ).join("")}</select>`).join("")
  }</div>`).join("");
 document.querySelectorAll("[data-rating]").forEach(node => {
  const value = ratings[node.dataset.rating];
  node.value = value == null ? "" : String(value);
  node.onchange = () => {
   ratings[node.dataset.rating] =
    node.value === "" ? null : Number(node.value);
   save();
  };
 });
}
function go(position) {
 if (position < 0 || position >= DATA.sections.length) return;
 index = position;
 render();
 save();
}
function restore(data) {
 if (!data || data.comparison_id !== DATA.id ||
     JSON.stringify(data.mapping) !== JSON.stringify(DATA.mapping) ||
     data.section_size !== DATA.section_size ||
     JSON.stringify(data.turn_counts) !== JSON.stringify(DATA.turn_counts)) {
  throw Error("Different comparison, source hashes, or section size");
 }
 if (!Array.isArray(data.sections) ||
     data.sections.length !== DATA.sections.length ||
     !Array.isArray(data.criteria) ||
     data.criteria.length !== DATA.criteria.length) {
  throw Error("Invalid review structure");
 }
 const v = {}, n = {}, r = {};
 const valid = [null,"A","B","tie","skip"];
 for (let i=0;i<DATA.sections.length;i++) {
  const row = data.sections[i], expected = DATA.sections[i];
  if (!row || row.id !== expected.id ||
      !valid.includes(row.preference ?? null) ||
      typeof row.note !== "string") {
   throw Error("Invalid section review");
  }
  v[row.id] = row.preference ?? null;
  n[row.id] = row.note;
 }
 for (let i=0;i<DATA.criteria.length;i++) {
  const item=data.criteria[i];
  if (!item || item.name !== DATA.criteria[i])
   throw Error("Criteria mismatch");
  for (const side of ["A","B"]) {
   const value=item[side];
   if (value != null &&
       (!Number.isInteger(value) || value < 1 || value > 5))
    throw Error("Invalid rating");
   r[i+":"+side]=value ?? null;
  }
 }
 if (!["in_progress","complete"].includes(data.status))
  throw Error("Invalid status");
 if (![null,"A","B","tie"].includes(data.overall_preference ?? null))
  throw Error("Invalid overall preference");
 if (typeof data.final_notes !== "string")
  throw Error("Invalid final notes");

 // No state changes until the whole import passes validation.
 for (const obj of [votes,notes,ratings])
  for (const key of Object.keys(obj)) delete obj[key];
 Object.assign(votes,v);
 Object.assign(notes,n);
 Object.assign(ratings,r);
 status=data.status;
 index=Math.max(0,DATA.sections.findIndex(
  s => s.id === data.current_section));
 el("status").value=status;
 el("overall").value=data.overall_preference ?? "";
 el("final-note").value=data.final_notes;
 renderRatings();
 render();
 save();
}
for (const s of DATA.sections) {
 const option=document.createElement("option");
 option.value=String(s.id);
 option.textContent="Section "+s.id;
 el("jump").appendChild(option);
}
el("prev").onclick=()=>go(index-1);
el("next").onclick=()=>go(index+1);
el("jump").onchange=e=>go(DATA.sections.findIndex(
 s=>s.id===Number(e.target.value)));
el("unrated").onclick=()=>{
 let next=DATA.sections.findIndex((s,i)=>
  i>index && !votes[s.id]);
 if(next<0) next=DATA.sections.findIndex(s=>!votes[s.id]);
 if(next>=0) go(next);
 else alert("No unrated sections remain.");
};
el("status").onchange=e=>{status=e.target.value;save()};
el("overall").onchange=save;
el("final-note").oninput=save;
el("export").onclick=()=>{
 const blob=new Blob([JSON.stringify(snapshot(),null,2)],
  {type:"application/json"});
 const a=document.createElement("a");
 a.href=URL.createObjectURL(blob);
 a.download="rpbench-conversation-"+DATA.id+".json";
 a.click();
 setTimeout(()=>URL.revokeObjectURL(a.href),2000);
};
el("import").onclick=()=>el("file").click();
el("file").onchange=async e=>{
 const file=e.target.files[0];
 if(!file) return;
 try {
  restore(JSON.parse(await file.text()));
  alert("Review restored.");
 } catch(err) {
  alert("Import failed: "+err.message);
 }
 e.target.value="";
};
renderRatings();
render();
try {
 const saved=localStorage.getItem(KEY);
 if(saved) restore(JSON.parse(saved));
 else save();
} catch(_) {
 el("storage").textContent="Autosave unavailable; use JSON export";
}
</script>
</body>
</html>"""


def export_html(data, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False)
    encoded = (
        encoded.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    output.write_text(
        PAGE.replace("__RPBENCH_DATA__", encoded),
        encoding="utf-8",
    )


def summarize(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = {"A": 0, "B": 0, "tie": 0, "skip": 0}
    for section in data["sections"]:
        vote = section.get("preference")
        if vote in counts:
            counts[vote] += 1
    rated = counts["A"] + counts["B"] + counts["tie"]
    return {
        "comparison_id": data["comparison_id"],
        "status": data.get("status", "in_progress"),
        "rated_sections": rated,
        "total_sections": len(data["sections"]),
        "votes": counts,
        "overall_preference": data.get("overall_preference"),
        "criteria": data.get("criteria", []),
        "mapping": data["mapping"],
        "final_notes": data.get("final_notes", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create")
    create.add_argument("--run-a", required=True)
    create.add_argument("--run-b", required=True)
    create.add_argument("--section-size", type=int, default=20)
    create.add_argument("--seed", default="personal-rp")
    create.add_argument("--out", required=True)

    report = sub.add_parser("report")
    report.add_argument("--results", required=True)

    args = parser.parse_args()
    if args.command == "create":
        data = prepare(
            args.run_a, args.run_b, args.section_size, args.seed
        )
        export_html(data, args.out)
        print("Created:", args.out)
        print("Sections:", len(data["sections"]))
        print("Turns:", data["turn_counts"])
    else:
        print(json.dumps(
            summarize(args.results), indent=2, ensure_ascii=False
        ))


if __name__ == "__main__":
    main()
