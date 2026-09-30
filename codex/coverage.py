#!/usr/bin/env python3
"""Count subset support on assistant output, without retokenizing input history."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regions(payload):
    kind = payload.get("type")
    if kind == "reasoning":
        # summary can repeat content; use content when available.
        for part in payload.get("content") or payload.get("summary", []):
            if part.get("text"):
                yield "reasoning", part["text"]
    elif kind == "message" and payload.get("role") == "assistant":
        for part in payload.get("content", []):
            if part.get("text"):
                yield "prose", part["text"]
    elif kind in ("function_call", "custom_tool_call"):
        text = payload.get("arguments", payload.get("input", ""))
        if text:
            yield "tool-call args", text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)


def items(path):
    seen = set()
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("type") == "response_item":
                payload = row["payload"]
                identity = payload.get("id")
                if identity and identity in seen:
                    continue
                if identity:
                    seen.add(identity)
                yield from regions(payload)
            elif path.name == "model-calls.jsonl":
                # Fail closed on an unknown logger schema instead of counting requests.
                response = row.get("response")
                if not isinstance(response, dict):
                    raise ValueError(f"{path}:{line_no}: response object missing")
                if "output" in response:
                    for payload in response["output"]:
                        yield from regions(payload)
                elif "choices" in response:
                    for choice in response["choices"]:
                        message = choice.get("message", {})
                        for key, region in (("reasoning_content", "reasoning"), ("content", "prose")):
                            if message.get(key):
                                yield region, message[key]
                        for call in message.get("tool_calls", []):
                            yield "tool-call args", call["function"]["arguments"]
                else:
                    raise ValueError(f"{path}:{line_no}: response output missing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / "repos/qwen38-eval/results/session2/pair-5060ti-x2")
    parser.add_argument("--tokenizer", type=Path, default=Path.home() / "models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/tokenizer.json")
    parser.add_argument("--subset", type=Path, default=Path.home() / "models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json")
    parser.add_argument("--gguf", type=Path, default=Path.home() / "models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "coverage.md")
    args = parser.parse_args()
    from tokenizers import Tokenizer
    start = time.monotonic()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "frozen/stack2/gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(args.gguf)
    gguf_tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    actual = tokenizer.get_vocab()
    assert all(gguf_tokens[token] == text for text, token in actual.items()), "tokenizer ID mapping differs from GGUF"
    actual_ids = set(actual.values())
    missing = [i for i in range(len(gguf_tokens)) if i not in actual_ids]
    assert all(gguf_tokens[i] == f"[PAD{i}]" for i in missing), "non-padding GGUF token missing from tokenizer"
    subset = set(json.loads(args.subset.read_text()))
    totals, outside = Counter(), defaultdict(Counter)
    cohorts = defaultdict(lambda: [0, 0])
    inputs = []
    logger_files = 0
    for campaign in sorted(args.root.glob("campaign-*swift15*")):
        calls = sorted(campaign.rglob("model-calls.jsonl"))
        logger_files += len(calls)
        # Llama campaigns retain rollouts rather than the HyperQwen logger.
        files = calls or sorted(campaign.glob("attempts/*/codex-state/sessions/**/rollout-*.jsonl"))
        for path in files:
            count = 0
            for region, text in items(path):
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                excluded = [token for token in ids if token not in subset]
                totals[region] += len(ids)
                outside[region].update(excluded)
                cohorts[campaign.name][0] += len(ids)
                cohorts[campaign.name][1] += len(excluded)
                count += len(ids)
            inputs.append({"path": str(path), "sha256": digest(path), "tokens": count})
    if not totals.total():
        raise SystemExit("no assistant output found")
    combined = sum(outside.values(), Counter())
    records = {"tokenizer": str(args.tokenizer), "tokenizer_sha256": digest(args.tokenizer),
               "subset": str(args.subset), "subset_sha256": digest(args.subset), "subset_size": len(subset),
               "inputs": inputs, "regions": dict(totals), "outside": dict(combined),
               "campaigns": dict(cohorts), "elapsed_s": time.monotonic() - start,
               "gguf": str(args.gguf), "matching_vocab_ids": len(actual), "gguf_padding_ids": len(missing)}
    args.out.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
    total, excluded = totals.total(), combined.total()
    lines = ["# Offline Swift draft-vocabulary coverage", "",
             f"M: {len(inputs)} input files across {len(cohorts)} campaigns; {logger_files} matching model-calls.jsonl files. "
             "Llama inputs use response_item records from Codex rollouts, including failed/retried attempts. "
             "Only assistant reasoning, prose, and tool arguments are counted; input history, event_msg copies, tool results, and tool names are excluded.", "",
             f"M: tokenizer SHA256 `{records['tokenizer_sha256']}`; subset SHA256 `{records['subset_sha256']}`; {len(subset)} unique IDs.", "",
             f"M: all {len(actual)} tokenizer IDs match IQ4_XS GGUF token spellings; the other {len(missing)} GGUF entries are [PAD...] placeholders absent from tokenizer.json.", "",
             "| Region | Tokens (M) | Outside subset (M) | Outside fraction (D) | Empirical acceptance ceiling (D) |",
             "|---|---:|---:|---:|---:|"]
    for region, n in [(r, totals[r]) for r in sorted(totals)] + [("all", total)]:
        miss = excluded if region == "all" else outside[region].total()
        lines.append(f"| {region} | {n} | {miss} | {miss/n:.6%} | {1-miss/n:.6%} |")
    lines += ["", "At any history, subset acceptance is at most p(S), since sum min(p,q) <= p(S). "
              "The outside-output fraction estimates the target mass forced into residual verification, so 1-f is an empirical ceiling, not a measured draft acceptance. "
              "Relative to a perfect full-vocabulary drafter, support alone forces loss f. Relative to the actual full drafter, incremental loss is unknown: renormalization also changes q within S. "
              "It is incorrect to claim f is an upper bound on total acceptance loss.", "",
              "Each recorded region is tokenized independently with tokenizer.json (NFC + byte BPE), without special-token insertion. "
              "These are reconstructed content tokens, not a capture of wire token IDs; message boundaries, template delimiters, parser stripping, and normalization can change counts. "
              "The sample is agent trajectories rather than independent draws. Full provenance is in coverage.json.", "",
              "| Outside token ID | Token spelling | Decoded piece | Count (M) |",
              "|---:|---|---|---:|"]
    for token, count in combined.most_common(30):
        spelling = json.dumps(tokenizer.id_to_token(token), ensure_ascii=True).replace("|", "\\|")
        decoded = json.dumps(tokenizer.decode([token], skip_special_tokens=False), ensure_ascii=True).replace("|", "\\|")
        lines.append(f"| {token} | `{spelling}` | `{decoded}` | {count} |")
    lines += ["", "| Campaign | Tokens (M) | Outside (M) | Fraction (D) |", "|---|---:|---:|---:|"]
    for campaign, (n, miss) in sorted(cohorts.items()):
        lines.append(f"| {campaign} | {n} | {miss} | {miss/n:.6%} |" if n else f"| {campaign} | 0 | 0 | unknown |")
    lines += ["", f"M: CPU elapsed {records['elapsed_s']:.3f} s; no inference or GPU access."]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"M: {total} tokens, {excluded} outside; D: {excluded/total:.6%}; M: {records['elapsed_s']:.3f} CPU wall s")


if __name__ == "__main__":
    main()
