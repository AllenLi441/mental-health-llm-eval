#!/usr/bin/env python3
"""Select an ESConv checkpoint only by the lowest recorded validation PPL."""

import argparse
import json
import math
import re
from pathlib import Path


NUMBER = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"


def parse_metrics(text):
    metrics = {}
    for key in ("perplexity", "ppl", "eval_ppl"):
        matches = re.findall(r"\b%s\b\s*[:=]\s*%s" % (key, NUMBER), text, re.I)
        if matches:
            metrics["ppl"] = float(matches[-1])
    matches = re.findall(r"\beval_loss\b\s*[:=]\s*%s" % NUMBER, text, re.I)
    if matches:
        metrics["eval_loss"] = float(matches[-1])
        metrics.setdefault("ppl", math.exp(metrics["eval_loss"]))
    return metrics


def checkpoint_candidates(output_dir):
    candidates = []
    for checkpoint in sorted(output_dir.glob("checkpoint-*")):
        texts = []
        for name in ("eval_results.txt", "eval_results.json", "trainer_state.json"):
            path = checkpoint / name
            if path.exists():
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
        metrics = parse_metrics("\n".join(texts))
        if "ppl" in metrics:
            candidates.append({"checkpoint": str(checkpoint.resolve()), **metrics})
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    args = parser.parse_args()

    candidates = checkpoint_candidates(args.output_dir)
    if not candidates:
        log_metrics = parse_metrics(args.training_log.read_text(encoding="utf-8", errors="replace"))
        if log_metrics:
            raise RuntimeError(
                "Validation metrics exist in the log but are not tied to checkpoint files; refusing to select the last checkpoint"
            )
        raise RuntimeError("No checkpoint with an auditable validation PPL was found")
    best = min(candidates, key=lambda item: item["ppl"])
    result = {
        "selection_rule": "lowest validation PPL",
        "best_checkpoint": best["checkpoint"],
        "best_validation_ppl": best["ppl"],
        "candidates": candidates,
    }
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(best["checkpoint"])


if __name__ == "__main__":
    main()
