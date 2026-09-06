#!/usr/bin/env python3
"""Prepare an auditable copy of the ACL 2021 official ``codes/`` runner."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "f262d062ad74cb39b17ea476facc81568ddcba24"
EXPECTED_DATASET_SHA256 = {
    "trainWithStrategy_short.tsv": "0ecf37462f8e3fa7f1dc5bfbecd70733abb3501c3bf83b27a957bec5a926ec21",
    "devWithStrategy_short.tsv": "625b511f40cf9a9285582e808e6bbb4c2f35d0b0cd063f3709db430b8d0c3bdc",
    "testWithStrategy_short.tsv": "b85ae888bf747cefa54bba2a6c3e2f6ccb4c1005d4e0b6d1d3be3823cf040aef",
}

CANONICAL_TOKENS = [
    "[Questions]",
    "[Reflection of feelings]",
    "[Information]",
    "[Restatement or Paraphrasing]",
    "[Other]",
    "[Self-disclosure]",
    "[Affirmation and Reassurance]",
    "[Providing Suggestions]",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_commit(repo):
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


ARGS_BLOCK = r'''
def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _audit_strategy_token_mapping(tokenizer):
    expected = {
        "[Questions]": 54944,
        "[Reflection of feelings]": 54945,
        "[Information]": 54946,
        "[Restatement or Paraphrasing]": 54947,
        "[Other]": 54948,
        "[Self-disclosure]": 54949,
        "[Affirmation and Reassurance]": 54950,
        "[Providing Suggestions]": 54951,
        "[CLS]": 54952,
    }
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in expected}
    if actual != expected:
        raise RuntimeError("strategy token mapping mismatch: expected=%r actual=%r" % (expected, actual))
    print("ESCONV_TOKEN_MAPPING=" + json.dumps(actual, sort_keys=True))


class Args():
    def __init__(self):
        self.output_dir = os.environ.get("ESCONV_OUTPUT_DIR", "./blender_strategy_repro_seed42")
        self.model_type = "mymodel"
        self.model_name_or_path = os.environ.get("ESCONV_MODEL_PATH", "./blender-small")
        self.config_name = os.environ.get("ESCONV_CONFIG_PATH", "./blender-small")
        self.tokenizer_name = os.environ.get("ESCONV_TOKENIZER_PATH", "./blender-small")
        self.data_path = os.environ.get("ESCONV_DATA_PATH", "./dataset")
        self.train_file_name = os.environ.get("ESCONV_TRAIN_FILE", "trainWithStrategy_short.tsv")
        self.eval_file_name = os.environ.get("ESCONV_EVAL_FILE", "devWithStrategy_short.tsv")
        self.cache_dir = os.environ.get("ESCONV_CACHE_DIR", "cached_repro_seed42")
        self.block_size = 512
        self.do_train = _env_bool("ESCONV_DO_TRAIN", True)
        self.do_eval = _env_bool("ESCONV_DO_EVAL", False)
        self.generation = _env_bool("ESCONV_GENERATION", False)
        self.generate_and_eval = _env_bool("ESCONV_GENERATE_AND_EVAL", False)
        self.evaluate_during_training = _env_bool("ESCONV_EVALUATE_DURING_TRAINING", True)
        self.per_gpu_train_batch_size = 4
        self.per_gpu_eval_batch_size = 1
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 5
        self.max_steps = -1
        self.warmup_steps = 120
        self.logging_steps = 30
        self.save_steps = 30
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = _env_bool("ESCONV_NO_CUDA", False)
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = int(os.environ.get("ESCONV_SEED", "42"))
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = "O1"
        self.strategy = True
        self.turn = False
        self.role = False
'''.strip("\n")


def patch_official_source(source):
    source, args_count = re.subn(
        r"class Args\(\):\n.*?(?=\nclass InputFeatures_train)",
        ARGS_BLOCK + "\n",
        source,
        flags=re.DOTALL,
    )
    if args_count != 1:
        raise RuntimeError("Could not replace the official Args block exactly once")

    canonical = "additional_special_tokens = " + repr(CANONICAL_TOKENS)
    source, token_list_count = re.subn(
        r"additional_special_tokens\s*=\s*\[[^\n]*\]", canonical, source
    )
    if token_list_count < 3:
        raise RuntimeError("Expected at least three official token-list assignments")

    audit_line = "    tokenizer.add_special_tokens({'cls_token': '[CLS]'})"
    audit_replacement = audit_line + "\n    _audit_strategy_token_mapping(tokenizer)"
    audit_count = source.count(audit_line)
    if audit_count < 3:
        raise RuntimeError("Expected at least three tokenizer initialization sites")
    source = source.replace(audit_line, audit_replacement)

    source = source.replace("args_do_eval", "args.do_eval")
    source = source.replace("BlenerbotSmallTokenizer", "BlenderbotSmallTokenizer")
    source = source.replace("wirte_path", "write_path")
    source = source.replace(
        'write_path = "./generated_data/generated_pretrained_single_p9t7k3rp3.json"',
        'write_path = os.environ.get("ESCONV_GENERATED_OUTPUT", "./generated_data/generated_pretrained_single_p9t7k3rp3.json")\n    os.makedirs(os.path.dirname(write_path), exist_ok=True)',
    )

    if '"[Question]"' in source or '"[Others]"' in source:
        raise RuntimeError("Forbidden strategy token spelling remains in patched source")
    if source.count("_audit_strategy_token_mapping(tokenizer)") < 3:
        raise RuntimeError("Token mapping audit was not injected at every runner entry point")
    return source


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--base-weight", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    commit = repo_commit(args.official_repo)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError("official repo commit mismatch: %s" % commit)
    data_dir = args.official_repo / "codes/dataset"
    dataset_hashes = {
        name: sha256_file(data_dir / name) for name in EXPECTED_DATASET_SHA256
    }
    if dataset_hashes != EXPECTED_DATASET_SHA256:
        raise RuntimeError("official dataset hash mismatch: %r" % dataset_hashes)
    if not args.base_weight.exists():
        raise FileNotFoundError(args.base_weight)

    if args.work_dir.exists():
        if not args.force:
            raise FileExistsError("work directory exists; pass --force to replace it")
        shutil.rmtree(str(args.work_dir))
    codes_target = args.work_dir / "codes"
    shutil.copytree(str(args.official_repo / "codes"), str(codes_target))
    target_weight = codes_target / "blender-small/pytorch_model.bin"
    shutil.copy2(str(args.base_weight), str(target_weight))

    runner = codes_target / "BlenderEmotionalSupport.py"
    original = runner.read_text(encoding="utf-8")
    original_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
    patched = patch_official_source(original)
    runner.write_text(patched, encoding="utf-8")
    patched_hash = hashlib.sha256(patched.encode("utf-8")).hexdigest()

    manifest = {
        "official_repo": str(args.official_repo.resolve()),
        "official_repo_commit": commit,
        "dataset_sha256": dataset_hashes,
        "base_model_weight": str(args.base_weight.resolve()),
        "base_model_sha256": sha256_file(args.base_weight),
        "copied_base_model_sha256": sha256_file(target_weight),
        "official_runner_sha256": original_hash,
        "patched_runner_sha256": patched_hash,
        "patches": [
            "environment-configurable Args with ACL 2021 defaults",
            "frozen checkpoint token order and runtime mapping assertion",
            "args.do_eval typo correction",
            "BlenderbotSmallTokenizer typo correction",
            "generation output path typo correction",
        ],
    }
    manifest_path = args.work_dir / "official_code_preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
