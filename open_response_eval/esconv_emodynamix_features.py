"""Verified history-only SDDP/ERC features for clean EmoDynamiX training.

The module imports only pinned upstream model classes, safely loads separately
hashed weights, and exposes no dataset or test split.  Labels and target
responses are neither accepted by the renderer nor written to feature rows.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


UPSTREAM_COMMIT = "c9213d718a9684a5e05ce5daa947f9cbbfb7b927"
FEATURE_SCHEMA_VERSION = "esconv-emodynamix-causal-feature-v1"
VERIFIED_FEATURE_BACKEND = "verified_upstream_sddp_erc_v1"
EXPECTED_ASSETS = {
    "erc_weights_sha256": (
        "5cf62bb54f97e09302b0c78dcdb2cdb2a7b1c96e761586b340746c0c8f614edf"
    ),
    "sddp_weights_sha256": (
        "746b9a7daed6f5679714f340f42f4b560cb42bf6fb81cc423282a1eda5d2f680"
    ),
    "sddp_config_sha256": (
        "ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9"
    ),
    "sddp_tokenizer_sha256": (
        "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e"
    ),
    "sddp_tree_sha256": (
        "f8eca6108f311cb079c8be6c510fce2977ac16a3eb6844f627bc291172b3aa22"
    ),
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_INPUT_FIELDS = {
    "dialogue_history",
    "strategy_history",
    "speaker_turn",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def render_upstream_feature_inputs(model_input: dict[str, Any]) -> dict[str, Any]:
    """Reproduce upstream preprocessing whitespace without accepting labels."""

    if set(model_input) != _MODEL_INPUT_FIELDS:
        raise ValueError("feature model input fields mismatch")
    dialogue_history = model_input["dialogue_history"]
    speaker_turn = model_input["speaker_turn"]
    if not isinstance(dialogue_history, str) or not isinstance(speaker_turn, str):
        raise ValueError("feature dialogue and speaker inputs must be strings")
    utterances = dialogue_history.split("</s>")
    speakers = speaker_turn.split(" ")
    if not 1 <= len(utterances) <= 5 or len(utterances) != len(speakers):
        raise ValueError("feature input node dimensions disagree")
    dialogue_for_parsing = [
        {"speaker": speaker, "text": utterance}
        for speaker, utterance in zip(speakers, utterances)
    ]
    erc_context = "".join(f" </s> {utterance}" for utterance in utterances)
    return {
        "dialogue_for_parsing": dialogue_for_parsing,
        "erc_context": erc_context,
        "node_count": len(utterances),
    }


def unique_causal_inputs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate feature work only; training records remain untouched."""

    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        try:
            key = record["model_input_sha256"]
            model_input = record["model_input"]
        except (KeyError, TypeError) as error:
            raise ValueError("record lacks causal model input fields") from error
        _require_sha256(key, "model input SHA")
        if canonical_json_sha256(model_input) != key:
            raise ValueError("model input hash does not match canonical input bytes")
        normalized = {
            "model_input_sha256": key,
            "model_input": model_input,
        }
        existing = by_key.get(key)
        if existing is not None and existing != normalized:
            raise ValueError("duplicate model input SHA maps to different content")
        by_key[key] = normalized
    if not by_key:
        raise ValueError("no causal model inputs were provided")
    return [by_key[key] for key in sorted(by_key)]


def build_generator_manifest(
    *,
    runtime_receipt: dict[str, Any],
    implementation_sha256: str,
    device: str,
) -> dict[str, Any]:
    _require_sha256(implementation_sha256, "feature implementation SHA")
    required_runtime = {
        "upstream_commit",
        "upstream_feature_code_tree_sha256",
        "base_model_tree_sha256",
        "sddp_tree_sha256",
        "sddp_weights_sha256",
        "erc_weights_sha256",
    }
    missing = sorted(required_runtime - set(runtime_receipt))
    if missing:
        raise ValueError(f"runtime receipt lacks required assets: {missing}")
    for field in required_runtime - {"upstream_commit"}:
        _require_sha256(runtime_receipt[field], field)
    if runtime_receipt["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("runtime receipt upstream commit mismatch")
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("feature device must be cpu, mps, or cuda")
    return {
        "schema_version": "esconv-emodynamix-feature-generator-v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_backend": VERIFIED_FEATURE_BACKEND,
        "implementation_sha256": implementation_sha256,
        "runtime_assets": runtime_receipt,
        "device": device,
        "causal_input_key": "model_input_sha256",
        "target_or_label_fields": [],
        "erc_cache_semantics": "upstream_post_softmax_probabilities",
        "author_double_softmax_preserved": True,
        "sddp_external_relation_range": [0, 16],
        "record_duplicates_preserved": True,
        "feature_keys_unique": True,
        "selectable_without_manifest_verification": False,
    }


def generator_manifest_sha256(manifest: dict[str, Any]) -> str:
    return canonical_json_sha256(manifest)


def _to_nested_float_lists(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("ERC runtime output is not a nested list")
    output: list[list[float]] = []
    for vector in value:
        if not isinstance(vector, list) or len(vector) != 7:
            raise ValueError("ERC runtime output must contain seven-way rows")
        numeric = [float(item) for item in vector]
        if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in numeric):
            raise ValueError("ERC runtime output contains invalid probabilities")
        if not math.isclose(sum(numeric), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("ERC runtime probabilities do not sum to one")
        output.append(numeric)
    return output


def generate_verified_feature_rows(
    input_rows: Sequence[dict[str, Any]],
    *,
    runtime: Any,
    generator_manifest_sha256: str,
    batch_size: int,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate deterministic feature rows from unique history-only inputs."""

    _require_sha256(generator_manifest_sha256, "generator manifest SHA")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("feature batch_size must be a positive integer")
    unique_inputs = unique_causal_inputs(input_rows)
    generated: list[dict[str, Any]] = []
    for start in range(0, len(unique_inputs), batch_size):
        batch = unique_inputs[start : start + batch_size]
        rendered = [render_upstream_feature_inputs(row["model_input"]) for row in batch]
        parsed_outputs = runtime.parse_batch(
            [item["dialogue_for_parsing"] for item in rendered]
        )
        erc_outputs = runtime.erc_batch(
            [item["erc_context"] for item in rendered],
            [item["node_count"] for item in rendered],
        )
        if len(parsed_outputs) != len(batch) or len(erc_outputs) != len(batch):
            raise ValueError("feature runtime returned the wrong batch length")
        for input_row, rendered_row, parsed, erc in zip(
            batch, rendered, parsed_outputs, erc_outputs
        ):
            edges: list[list[int]] = []
            for edge in parsed:
                if not isinstance(edge, (list, tuple)) or len(edge) != 3:
                    raise ValueError("SDDP runtime returned a malformed edge")
                head, tail, relation = edge
                if any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in (head, tail, relation)
                ):
                    raise ValueError("SDDP runtime edge values must be integers")
                node_count = rendered_row["node_count"]
                if not 0 <= head <= node_count or not 1 <= tail <= node_count:
                    raise ValueError("SDDP runtime edge endpoint is out of range")
                if not 0 <= relation <= 16:
                    raise ValueError("SDDP runtime relation is out of range")
                edges.append([head, tail, relation])
            edges = sorted(edges)
            if len(edges) != len({tuple(edge) for edge in edges}):
                raise ValueError("SDDP runtime returned duplicate edges")
            probabilities = _to_nested_float_lists(erc)
            if len(probabilities) != rendered_row["node_count"]:
                raise ValueError("ERC runtime node count does not match causal input")
            generated.append(
                {
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "model_input_sha256": input_row["model_input_sha256"],
                    "node_count": rendered_row["node_count"],
                    "parsed_dialogue": edges,
                    "upstream_erc_softmax_output": probabilities,
                    "feature_backend": VERIFIED_FEATURE_BACKEND,
                    "generator_manifest_sha256": generator_manifest_sha256,
                }
            )
        if progress_callback is not None:
            progress_callback(
                {"completed": len(generated), "total": len(unique_inputs)}
            )
    return sorted(generated, key=lambda row: row["model_input_sha256"])


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_upstream_worktree(repo: Path, paths: Sequence[str]) -> None:
    tracked = subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", *paths],
        check=False,
    )
    if tracked.returncode not in {0, 1}:
        raise ValueError("could not audit upstream feature implementation")
    if tracked.returncode == 1:
        raise ValueError("upstream feature implementation has tracked changes")
    untracked_output = _git_output(
        repo, "ls-files", "--others", "--exclude-standard", "--", *paths
    )
    unexpected = []
    for value in untracked_output.splitlines():
        path = Path(value)
        if "__pycache__" in path.parts and path.suffix == ".pyc":
            continue
        unexpected.append(value)
    if unexpected:
        raise ValueError(
            f"upstream feature implementation has untracked files: {unexpected}"
        )


def _hash_code_tree(root: Path, paths: Sequence[str]) -> str:
    entries: list[dict[str, Any]] = []
    for relative_root in paths:
        candidate = root / relative_root
        if candidate.is_symlink() or not candidate.exists():
            raise ValueError(f"upstream feature code path is missing or symlinked: {candidate}")
        files = [candidate] if candidate.is_file() else list(candidate.rglob("*.py"))
        for path in sorted(files, key=lambda item: item.as_posix()):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"upstream feature code contains a symlink: {path}")
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return canonical_json_sha256(entries)


def _hash_recovery_asset_tree(root: Path) -> str:
    """Match the recovery receipt's sorted ``shasum`` tree commitment."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("asset tree is missing or symlinked")
    lines: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"asset tree contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            lines.append(f"{sha256_file(path)}  ./{relative}\n")
    if not lines:
        raise ValueError("asset tree contains no files")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


class _VerifiedFeatureRuntime:
    def __init__(self, parser: Any, erc: Any, device: Any) -> None:
        self.parser = parser
        self.erc = erc
        self.device = device

    def parse_batch(self, dialogues: Sequence[list[dict[str, str]]]) -> list[Any]:
        return self.parser.parse(list(dialogues))

    def erc_batch(
        self, contexts: Sequence[str], node_counts: Sequence[int]
    ) -> list[list[list[float]]]:
        import torch

        with torch.no_grad():
            flattened = self.erc({"texts": list(contexts)})["logits"]
        if flattened.shape != (sum(node_counts), 7):
            raise ValueError(
                "ERC runtime flattened output shape does not match requested nodes"
            )
        outputs: list[list[list[float]]] = []
        offset = 0
        for count in node_counts:
            outputs.append(
                flattened[offset : offset + count].detach().cpu().tolist()
            )
            offset += count
        return outputs


def _import_pinned_upstream(upstream_repo: Path) -> tuple[Any, Any, Any]:
    repo_string = str(upstream_repo)
    existing = sys.modules.get("modules")
    if existing is not None:
        origin = Path(getattr(existing, "__file__", "")).resolve()
        if upstream_repo.resolve() not in origin.parents:
            raise ValueError("a different top-level modules package is already imported")
    sys.path.insert(0, repo_string)
    try:
        sddp_module = importlib.import_module("modules.sddp.sddp")
        erc_module = importlib.import_module("modules.erc.sequential")
        decoder_module = importlib.import_module("modules.decoder")
    finally:
        try:
            sys.path.remove(repo_string)
        except ValueError:
            pass
    return sddp_module, erc_module, decoder_module


def load_verified_feature_runtime(
    *,
    upstream_repo: Path,
    asset_root: Path,
    base_model_path: Path,
    expected_base_tree_sha256: str,
    device: str,
) -> tuple[_VerifiedFeatureRuntime, dict[str, Any]]:
    """Safely instantiate pinned SDDP/ERC without upstream dataset entrypoints."""

    import torch
    from torch import nn
    import torch_geometric
    import torch_struct
    import transformers
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

    from open_response_eval.esconv_emodynamix_model import _validate_base_model_tree

    upstream_repo = Path(upstream_repo)
    asset_root = Path(asset_root)
    base_model_path = Path(base_model_path)
    if any(path.is_symlink() for path in (upstream_repo, asset_root, base_model_path)):
        raise ValueError("runtime roots must not be symlinks")
    if not upstream_repo.is_dir() or not asset_root.is_dir():
        raise ValueError("runtime upstream repository or asset root is missing")
    head = _git_output(upstream_repo, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        raise ValueError(f"upstream repository commit mismatch: {head}")
    relevant_paths = ["modules/sddp", "modules/erc", "modules/decoder"]
    _validate_upstream_worktree(upstream_repo, relevant_paths)
    code_tree_sha = _hash_code_tree(upstream_repo, relevant_paths)

    base_commitment, _base_entries = _validate_base_model_tree(
        base_model_path, expected_base_tree_sha256
    )
    sddp_root = asset_root / "sddp_stac"
    sddp_weights = sddp_root / "pytorch_model.bin"
    sddp_config = sddp_root / "config.json"
    sddp_tokenizer = sddp_root / "tokenizer.json"
    erc_weights = asset_root / "sequential_erc_model.pth"
    required = [sddp_weights, sddp_config, sddp_tokenizer, erc_weights]
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise ValueError("required SDDP/ERC asset is missing or symlinked")
    actual_assets = {
        "erc_weights_sha256": sha256_file(erc_weights),
        "sddp_weights_sha256": sha256_file(sddp_weights),
        "sddp_config_sha256": sha256_file(sddp_config),
        "sddp_tokenizer_sha256": sha256_file(sddp_tokenizer),
        "sddp_tree_sha256": _hash_recovery_asset_tree(sddp_root),
    }
    drift = {
        key: {"expected": EXPECTED_ASSETS[key], "actual": value}
        for key, value in actual_assets.items()
        if value != EXPECTED_ASSETS[key]
    }
    if drift:
        raise ValueError(f"SDDP/ERC asset hash mismatch: {json.dumps(drift, sort_keys=True)}")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS feature device is unavailable")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA feature device is unavailable")
    elif device != "cpu":
        raise ValueError("feature device must be cpu, mps, or cuda")
    torch_device = torch.device(device)

    sddp_module, erc_module, decoder_module = _import_pinned_upstream(
        upstream_repo
    )
    parser = sddp_module.StructuredDialogueDiscourseParser.__new__(
        sddp_module.StructuredDialogueDiscourseParser
    )
    parser.device = torch_device
    parser.max_contexts_length = 48
    # Five is the maximum clean canonical history. This avoids constructing
    # dummy pairs for the upstream general-purpose limit of 37.
    parser.max_num_contexts = 5
    parser.tokenizer = sddp_module.AutoTokenizer.from_pretrained(
        str(sddp_root), local_files_only=True
    )
    encoder_config = sddp_module.AutoConfig.from_pretrained(
        str(sddp_config), local_files_only=True
    )
    encoder = sddp_module.AutoModel.from_config(encoder_config)
    parser.model = sddp_module.Model(
        encoder_config, encoder=encoder, link_only=False
    )
    sddp_state = torch.load(
        sddp_weights,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    position_ids_key = "encoder.embeddings.position_ids"
    if position_ids_key not in sddp_state:
        raise ValueError("SDDP checkpoint lacks expected legacy position_ids buffer")
    position_ids = sddp_state.pop(position_ids_key)
    expected_position_ids = torch.arange(
        position_ids.shape[-1], dtype=position_ids.dtype
    ).unsqueeze(0)
    if tuple(position_ids.shape) != (1, 514) or not torch.equal(
        position_ids.cpu(), expected_position_ids
    ):
        raise ValueError("SDDP legacy position_ids buffer has unexpected content")
    sddp_incompatible = parser.model.load_state_dict(sddp_state, strict=True)
    parser.model.to(torch_device)
    parser.model.eval()

    erc = erc_module.SequentialERC.__new__(erc_module.SequentialERC)
    nn.Module.__init__(erc)
    erc.device = torch_device
    erc.tokenizer = RobertaTokenizer.from_pretrained(
        str(base_model_path), local_files_only=True
    )
    erc.roberta_config = RobertaConfig.from_pretrained(
        str(base_model_path), local_files_only=True
    )
    # Every encoder tensor is replaced by the strict ERC state below; construct
    # from pinned config to avoid an unnecessary 500 MB base-weight load.
    erc.encoder = RobertaModel(erc.roberta_config)
    erc.classifier = decoder_module.RobertaClassificationHead(
        hidden_size=erc.roberta_config.hidden_size,
        num_labels=7,
    )
    erc.softmax = nn.Softmax(dim=-1)
    erc_state = torch.load(
        erc_weights,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    erc_incompatible = erc.load_state_dict(erc_state, strict=True)
    erc.to(torch_device)
    erc.device = torch_device
    erc.eval()

    runtime = _VerifiedFeatureRuntime(parser, erc, torch_device)
    torch_struct_root = Path(torch_struct.__file__).resolve().parent
    receipt = {
        "upstream_commit": head,
        "upstream_feature_code_tree_sha256": code_tree_sha,
        "base_model_tree_sha256": base_commitment,
        "sddp_tree_sha256": EXPECTED_ASSETS["sddp_tree_sha256"],
        **actual_assets,
        "device": device,
        "sddp_max_contexts_length": 48,
        "sddp_max_num_contexts": 5,
        "safe_weights_only": True,
        "mmap": True,
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_geometric": torch_geometric.__version__,
            "torch_struct": getattr(torch_struct, "version", "UNKNOWN"),
            "torch_struct_origin": str(torch_struct_root),
            "torch_struct_python_tree_sha256": _hash_code_tree(
                torch_struct_root, ["."]
            ),
        },
        "sddp_load_missing_keys": list(sddp_incompatible.missing_keys),
        "sddp_load_unexpected_keys": list(sddp_incompatible.unexpected_keys),
        "sddp_legacy_position_ids_verified_and_removed": True,
        "erc_load_missing_keys": list(erc_incompatible.missing_keys),
        "erc_load_unexpected_keys": list(erc_incompatible.unexpected_keys),
        "test_dataset_loaded": False,
    }
    return runtime, receipt
