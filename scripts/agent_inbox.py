"""Work the agent bridge inbox: the Cursor project agent (or a processor it runs) uses this to see pending
stage requests and write answers.

    python scripts/agent_inbox.py list                      # pending requests (inbox without an answer)
    python scripts/agent_inbox.py show <id>                 # print the system prompt, the user prompt, nothing else
    python scripts/agent_inbox.py wait [--timeout 540]      # block until a request is pending, then `show` the oldest
    python scripts/agent_inbox.py answer <id> --file a.json # validate and write the answer atomically
    python scripts/agent_inbox.py answer <id> < a.json
    python scripts/agent_inbox.py drop <id>                 # discard a request (no answer will come)

The bridge directory comes from AVATAR_AGENT_BRIDGE_DIR (default data/agent-bridge) or --dir.
Exit codes: 0 ok, 2 bad input, 3 nothing pending before the timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avatar.llm.agent_bridge import INBOX, OUTBOX, pending_requests, read_answer, write_json_atomic  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_list(base: Path, _args) -> int:
    pend = pending_requests(base)
    if not pend:
        print("no pending requests")
        return 0
    now = time.time()
    for p in pend:
        req = _load(p)
        print(f"{req['id']}\t{req['stage']}\tage {int(now - p.stat().st_mtime)}s")
    return 0


def cmd_show(base: Path, args) -> int:
    p = base / INBOX / f"{args.id}.json"
    if not p.exists():
        print(f"no such request: {args.id}", file=sys.stderr)
        return 2
    req = _load(p)
    print(f"=== REQUEST {req['id']}  stage={req['stage']}  created={req['created_at']}")
    print(f"=== ANSWER FILE: {base / OUTBOX / (req['id'] + '.json')}")
    print("=== SYSTEM PROMPT")
    print(req["system"])
    print("=== USER PROMPT")
    print(req["user"])
    return 0


def cmd_wait(base: Path, args) -> int:
    deadline = time.time() + args.timeout
    while True:
        pend = pending_requests(base)
        if pend:
            print(f"{len(pend)} pending request(s): " + ", ".join(p.stem for p in pend))
            args.id = pend[0].stem
            return cmd_show(base, args)
        if time.time() > deadline:
            print("no pending requests before timeout")
            return 3
        time.sleep(args.poll)


def cmd_answer(base: Path, args) -> int:
    p = base / INBOX / f"{args.id}.json"
    if not p.exists():
        print(f"no such request: {args.id}", file=sys.stderr)
        return 2
    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.startswith("json") else text
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"answer is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(obj, dict):
        print("answer must be a JSON object", file=sys.stderr)
        return 2
    out = base / OUTBOX / f"{args.id}.json"
    write_json_atomic(out, obj)
    assert read_answer(out) is not None
    print(f"answered {args.id} -> {out}")
    return 0


def cmd_drop(base: Path, args) -> int:
    p = base / INBOX / f"{args.id}.json"
    if p.exists():
        p.unlink()
        print(f"dropped {args.id}")
        return 0
    print(f"no such request: {args.id}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=os.environ.get("AVATAR_AGENT_BRIDGE_DIR", "data/agent-bridge"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("id")
    w = sub.add_parser("wait")
    w.add_argument("--timeout", type=float, default=540)
    w.add_argument("--poll", type=float, default=2.0)
    a = sub.add_parser("answer")
    a.add_argument("id")
    a.add_argument("--file")
    d = sub.add_parser("drop")
    d.add_argument("id")
    args = parser.parse_args()
    base = Path(args.dir)
    (base / INBOX).mkdir(parents=True, exist_ok=True)
    (base / OUTBOX).mkdir(parents=True, exist_ok=True)
    return {"list": cmd_list, "show": cmd_show, "wait": cmd_wait, "answer": cmd_answer, "drop": cmd_drop}[args.cmd](base, args)


if __name__ == "__main__":
    sys.exit(main())
