"""Extract visible messages from a downloaded ChatGPT share, without executing HTML.

The original HTML stays in the ignored private reference library. The output
records source provenance and preserves messages as research input, not evidence
that any reported simulation has been reproduced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path


def extract(html: str) -> dict:
    match = re.search(r'streamController.enqueue\((".*?")\);', html, re.DOTALL)
    if match is None:
        raise ValueError("No serialized shared conversation found.")
    values = json.loads(json.loads(match.group(1)))
    cache = {}

    def decode(index):
        if index < 0:
            return None
        if index in cache:
            return cache[index]
        value = values[index]
        if isinstance(value, dict):
            result = {}
            cache[index] = result
            result.update({str(decode(int(k[1:]))): decode(v) for k, v in value.items()})
            return result
        if isinstance(value, list):
            result = []
            cache[index] = result
            result.extend(decode(v) if isinstance(v, int) else v for v in value)
            return result
        return value

    root = decode(0)
    route = root["loaderData"]["routes/share.$shareId.($action)"]
    conversation = route["serverResponse"]["data"]
    messages = []
    for index, node in enumerate(conversation["linear_conversation"]):
        msg = node.get("message") or {}
        role = msg.get("author", {}).get("role")
        channel = msg.get("channel")
        parts = msg.get("content", {}).get("parts", [])
        content = "\n".join(part for part in parts if isinstance(part, str))
        if content and (role == "user" or (role == "assistant" and channel in (None, "final"))):
            messages.append({"index": index, "role": role, "content": content})
    return {
        "title": conversation["title"],
        "messages": messages,
        "links": conversation.get("safe_urls", []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = extract(args.html.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    for message in data.pop("messages"):
        (args.out / f"{message['index']:03d}-{message['role']}.md").write_text(
            message["content"] + "\n"
        )
    data.update(
        {
            "source_url": args.url,
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "html_sha256": hashlib.sha256(args.html.read_bytes()).hexdigest(),
            "status": "User-supplied research discussion; numerical claims not independently reproduced.",
        }
    )
    (args.out / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
