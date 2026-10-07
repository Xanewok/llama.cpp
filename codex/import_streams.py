#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def recorded_stream(path):
    data = json.loads(path.read_text())
    prompt = data["request"]["prompt"]
    tokens = []
    for chunk in data["chunks"]:
        if not isinstance(chunk, dict):
            continue
        ids = chunk.get("tokens")
        if isinstance(ids, list):
            tokens.extend(ids)
    if not prompt or not tokens or any(type(x) is not int or x < 0 for x in prompt + tokens):
        raise ValueError(f"{path}: prompt/generated ids missing or invalid")
    return {"id": path.stem, "prompt": prompt, "continuation": tokens}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    args = p.parse_args()
    streams = [recorded_stream(path) for path in args.paths]
    if args.canary:
        streams = streams[:1]
        streams[0]["continuation"] = streams[0]["continuation"][:16]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(streams) + "\n")
    print(f"streams_M={len(streams)} tokens_M={sum(len(s['prompt']) + len(s['continuation']) for s in streams)}")


if __name__ == "__main__":
    main()
