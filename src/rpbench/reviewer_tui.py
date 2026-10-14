"""Terminal reviewer for archived RPBench conversations."""

from __future__ import annotations

import argparse
import curses
import json
import textwrap
from pathlib import Path

from .review import (
    EVALUATION_TYPES,
    ALL_TYPES,
    observations,
    proposals,
    decisions,
    propose,
    decide,
)


def load_turns(run_dir):
    result = {}
    for entry in observations(run_dir):
        turn = entry["turn"]
        role = entry["role"]
        if role not in {"player", "npc"}:
            continue
        if role in result.setdefault(turn, {}):
            raise ValueError(
                f"Duplicate {role} observation at turn {turn}"
            )
        result[turn][role] = entry
    return result


def wrap(text, width):
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(
                paragraph,
                width=max(1, width),
                replace_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""])
    return lines or [""]


def build_display(turn, pair, show_reasoning=False, events=None):
    lines = [f"TURN {turn}", ""]

    for role in ("player", "npc"):
        observation = pair.get(role)
        if observation is None:
            continue

        lines.extend([
            f"=== {role.upper()} ===", ""
        ])
        lines.extend(
            observation["text"].splitlines()
        )
        lines.append("")

        if show_reasoning:
            matching = [
                event for event in events or []
                if event.get("type") == "response"
                and event.get("turn") == turn
                and event.get("role") == role
            ]
            if matching:
                reasoning = matching[-1].get("reasoning")
                if reasoning:
                    lines.extend([
                        f"--- {role.upper()} REASONING ---",
                        reasoning,
                        "",
                    ])

    return "\n".join(lines)


def read_events(run_dir):
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def prompt_input(stdscr, label, default=""):
    height, width = stdscr.getmaxyx()
    if height < 8 or width < 45:
        return None

    curses.echo()
    try:
        stdscr.move(height - 2, 0)
        stdscr.clrtoeol()
        stdscr.addnstr(
            height - 2, 0,
            f"{label}: ", width - 1,
        )
        value = stdscr.getstr(
            height - 2,
            min(len(label) + 2, width - 2),
            max(1, width - len(label) - 4),
        ).decode("utf-8")
        return value if value else default
    finally:
        curses.noecho()


class Reviewer:
    def __init__(self, run_dir, reviewer):
        self.run_dir = Path(run_dir)
        self.reviewer = reviewer
        self.turns = load_turns(run_dir)
        if not self.turns:
            raise ValueError("No archived observations found")

        self.turn_ids = sorted(self.turns)
        self.position = 0
        self.scroll = 0
        self.role = "npc"
        self.reasoning = False
        self.message = "Ready"
        self.events = read_events(run_dir)

    def current_turn(self):
        return self.turn_ids[self.position]

    def current_observation(self):
        return self.turns[
            self.current_turn()
        ].get(self.role)

    def move(self, delta):
        self.position = max(
            0,
            min(len(self.turn_ids) - 1, self.position + delta)
        )
        self.scroll = 0

    def make_proposal(self, stdscr):
        observation = self.current_observation()
        if observation is None:
            self.message = "No observation for this role"
            return

        kinds = sorted(ALL_TYPES)
        kind = prompt_input(
            stdscr,
            "Type (" + ",".join(kinds) + ")",
        )
        if not kind:
            self.message = "Cancelled"
            return

        if kind not in ALL_TYPES:
            self.message = "Unknown proposal type"
            return

        quote = prompt_input(
            stdscr, "Exact evidence quote"
        )
        rationale = prompt_input(
            stdscr, "Rationale"
        )

        if not quote or not rationale:
            self.message = "Cancelled: quote and rationale required"
            return

        payload = {}
        if kind not in EVALUATION_TYPES:
            raw = prompt_input(
                stdscr,
                "State payload JSON",
                "{}",
            )
            if raw is None:
                self.message = "Cancelled"
                return
            payload = json.loads(raw)

        item = propose(
            self.run_dir,
            kind,
            self.current_turn(),
            self.role,
            quote,
            rationale,
            payload,
        )
        self.message = f"Created proposal #{item['id']}"

    def show_proposals(self, stdscr):
        items = proposals(self.run_dir)
        completed = {
            d["proposal_id"]: d
            for d in decisions(self.run_dir)
        }
        self.message = (
            f"Proposals: {len(items)} total, "
            f"{len(completed)} decided"
        )

        if not items:
            return

        last = items[-1]
        decision = completed.get(last["id"])
        status = (
            decision["decision"] if decision
            else "pending"
        )
        self.message += (
            f" | latest #{last['id']}: "
            f"{last['type']} ({status})"
        )

    def review_decision(self, stdscr, decision):
        identifier = prompt_input(
            stdscr, "Proposal ID"
        )
        if not identifier:
            self.message = "Cancelled"
            return

        note = prompt_input(
            stdscr, "Review note", ""
        )
        if note is None:
            self.message = "Cancelled"
            return

        entry = decide(
            self.run_dir,
            int(identifier),
            decision,
            self.reviewer,
            note,
        )
        self.message = (
            f"Proposal #{entry['proposal_id']}: "
            f"{entry['decision']}"
        )

    def draw(self, stdscr):
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 8 or width < 45:
            stdscr.addstr(
                0, 0, "Terminal too small (min 45x8)"
            )
            stdscr.refresh()
            return

        turn = self.current_turn()
        header = (
            f"RPBench Review | Turn {turn} "
            f"({self.position + 1}/{len(self.turn_ids)}) "
            f"| Selected: {self.role}"
        )
        stdscr.addnstr(
            0, 0, header, width - 1,
            curses.A_REVERSE,
        )

        content = build_display(
            turn,
            self.turns[turn],
            self.reasoning,
            self.events,
        )
        display_lines = wrap(content, width - 2)
        view_height = height - 5
        max_scroll = max(
            0, len(display_lines) - view_height
        )
        self.scroll = min(self.scroll, max_scroll)

        visible = display_lines[
            self.scroll:self.scroll + view_height
        ]
        for index, line in enumerate(visible, 1):
            stdscr.addnstr(
                index, 0, line, width - 1
            )

        stdscr.addnstr(
            height - 4, 0,
            f"Scroll {self.scroll}/{max_scroll} "
            f"| Reasoning {'on' if self.reasoning else 'off'}",
            width - 1,
        )
        stdscr.addnstr(
            height - 3, 0,
            "Left/Right: turns | Up/Down: scroll "
            "| Tab: role | r: reasoning",
            width - 1,
        )
        stdscr.addnstr(
            height - 2, 0,
            "p: propose | v: proposals | "
            "a: accept | x: reject | q: quit",
            width - 1,
        )
        stdscr.addnstr(
            height - 1, 0,
            self.message,
            width - 1,
            curses.A_BOLD,
        )
        stdscr.refresh()

    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)

        while True:
            self.draw(stdscr)
            key = stdscr.getch()

            try:
                if key in (ord("q"), 27):
                    return
                elif key == curses.KEY_RIGHT:
                    self.move(1)
                elif key == curses.KEY_LEFT:
                    self.move(-1)
                elif key == curses.KEY_UP:
                    self.scroll = max(0, self.scroll - 1)
                elif key == curses.KEY_DOWN:
                    self.scroll += 1
                elif key == curses.KEY_NPAGE:
                    self.scroll += max(
                        1, stdscr.getmaxyx()[0] - 6
                    )
                elif key == curses.KEY_PPAGE:
                    self.scroll = max(
                        0,
                        self.scroll -
                        max(1, stdscr.getmaxyx()[0] - 6),
                    )
                elif key == 9:
                    self.role = (
                        "player" if self.role == "npc"
                        else "npc"
                    )
                elif key == ord("r"):
                    self.reasoning = not self.reasoning
                    self.scroll = 0
                elif key == ord("p"):
                    self.make_proposal(stdscr)
                elif key == ord("v"):
                    self.show_proposals(stdscr)
                elif key == ord("a"):
                    self.review_decision(stdscr, "accept")
                elif key == ord("x"):
                    self.review_decision(stdscr, "reject")
            except (ValueError, KeyError, OSError) as exc:
                self.message = f"Error: {exc}"


def main():
    parser = argparse.ArgumentParser(
        description="RPBench terminal review"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--reviewer", default="local-human"
    )
    args = parser.parse_args()

    reviewer = Reviewer(
        args.run, args.reviewer
    )
    curses.wrapper(reviewer.run)


if __name__ == "__main__":
    main()
