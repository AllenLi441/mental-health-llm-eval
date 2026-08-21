"""Clean, train/dev-only EmoDynamiX strategy-policy model.

This is a narrow derivative of the MIT-licensed ``cw-wan/EmoDynamiX-v2``
lightmode architecture at commit c9213d718a9684a5e05ce5daa947f9cbbfb7b927.
It intentionally contains no SDDP/ERC loaders, dataset paths, trainer, or test
hook.  Discourse and emotion features must already satisfy the independent
history-only feature contract.

The author-compatible control preserves the published computation quirks:
ERC probabilities receive a second softmax, ``<START>`` receives strategy 0,
root SDDP edges are discarded, and the position/FFN branches remain unused.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch_geometric.nn.conv import RGATConv
from transformers import RobertaModel, RobertaTokenizer


EXPECTED_BASE_TREE_SHA256 = (
    "1d9faa93557a63a92292cd11dfbca3de8e336ffa60768745a71ecd1ed19aa91c"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_TREE_IGNORED_PARTS = {".git", ".cache", "__pycache__"}
_MODEL_TREE_IGNORED_NAMES = {".DS_Store"}
_MODEL_INPUT_FIELDS = {
    "dialogue_history",
    "strategy_history",
    "speaker_turn",
}
_MINIMAL_FEATURE_FIELDS = {
    "node_count",
    "parsed_dialogue",
    "upstream_erc_softmax_output",
}
_OPTIONAL_FEATURE_FIELDS = {
    "schema_version",
    "model_input_sha256",
    "feature_backend",
    "generator_manifest_sha256",
}


@dataclass(frozen=True)
class CleanEmoDynamiXConfig:
    upstream_repository: str = "https://github.com/cw-wan/EmoDynamiX-v2"
    upstream_commit: str = "c9213d718a9684a5e05ce5daa947f9cbbfb7b927"
    base_hidden_size: int = 768
    graph_dim: int = 512
    erc_classes: int = 7
    strategy_classes: int = 8
    relations: int = 19
    graph_layers: int = 3
    erc_temperature: float = 0.5
    temperature_scalar: float = 100.0
    erc_mixed: bool = True
    author_double_softmax: bool = True
    classifier_dropout: float = 0.1
    max_context_tokens: int = 512


def author_context_string(dialogue_history: str, speakers: Sequence[str]) -> str:
    """Render the exact whitespace behavior of the published lightmode code."""

    utterances = dialogue_history.split("</s>")
    if len(utterances) != len(speakers):
        raise ValueError("dialogue and speaker lengths disagree")
    return " ".join(
        f"[{speaker}] {utterance}"
        for speaker, utterance in zip(speakers, utterances)
    )


class _UnusedFFN(nn.Module):
    """Checkpoint-compatible branch intentionally unused by author forward."""

    def __init__(self, dim_in: int, dim_out: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_in // 2, dim_out)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.linear(values)
        values = self.relu(values)
        values = self.dropout(values)
        return self.linear2(values)


class _AuthorGATLayer(nn.Module):
    def __init__(self, graph_dim: int, relations: int) -> None:
        super().__init__()
        self.conv = RGATConv(
            in_channels=graph_dim,
            out_channels=graph_dim,
            num_relations=relations,
        )
        self.ffn = _UnusedFFN(graph_dim, graph_dim, dropout=0.2)

    def forward(
        self,
        values: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        residual = values
        values, attention = self.conv(
            values,
            edge_index,
            edge_type,
            return_attention_weights=True,
        )
        return values + residual, attention


class _AuthorClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, labels: int, dropout: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, labels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.dropout(values)
        values = self.dense(values)
        values = torch.tanh(values)
        values = self.dropout(values)
        return self.out_proj(values)


class CleanEmoDynamiXPolicy(nn.Module):
    """Faithful lightmode graph policy with dynamic device handling."""

    def __init__(
        self,
        roberta: RobertaModel,
        tokenizer: RobertaTokenizer,
        config: CleanEmoDynamiXConfig,
    ) -> None:
        super().__init__()
        if roberta.config.hidden_size != config.base_hidden_size:
            raise ValueError(
                "RoBERTa hidden size does not match frozen EmoDynamiX architecture"
            )
        self.config_contract = config
        self.roberta = roberta
        self.roberta_tokenizer = tokenizer

        self.erc_prototypes = nn.Parameter(
            torch.randn(config.erc_classes, config.graph_dim)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.scalar = config.temperature_scalar
        self.t = nn.Parameter(
            torch.tensor(config.erc_temperature / config.temperature_scalar)
        )

        self.conv1 = _AuthorGATLayer(config.graph_dim, config.relations)
        self.conv2 = _AuthorGATLayer(config.graph_dim, config.relations)
        self.conv3 = _AuthorGATLayer(config.graph_dim, config.relations)
        self.dummy_embedding = nn.Parameter(torch.randn(config.graph_dim))
        self.strategy_embedding = nn.Embedding(
            config.strategy_classes, config.graph_dim
        )
        # Present for exact author architecture; author forward computes but
        # never adds these embeddings to graph nodes.
        self.node_position_embedding = nn.Embedding(6, config.graph_dim)
        self.classifier = _AuthorClassificationHead(
            config.base_hidden_size + config.graph_dim,
            config.strategy_classes,
            config.classifier_dropout,
        )

    @property
    def device(self) -> torch.device:
        # Unlike upstream's cached CUDA/CPU value, this follows model.to(...).
        return next(self.parameters()).device

    def _encode_contexts(self, contexts: Sequence[str]) -> torch.Tensor:
        tokenized = self.roberta_tokenizer(
            list(contexts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        if tokenized["input_ids"].shape[1] > self.config_contract.max_context_tokens:
            raise ValueError("context exceeds frozen no-truncation token limit")
        outputs = self.roberta(
            input_ids=tokenized["input_ids"].to(self.device),
            attention_mask=tokenized["attention_mask"].to(self.device),
        )
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, samples: dict[str, Any]) -> dict[str, Any]:
        expected_fields = {
            "dialogue_history",
            "strategy_history",
            "speaker_turn",
            "parsed_dialogue",
            "erc_probabilities",
            "dialogue_sizes",
        }
        if set(samples) != expected_fields:
            raise ValueError("clean model batch fields mismatch")
        batch_size = len(samples["dialogue_history"])
        if not (
            len(samples["strategy_history"])
            == len(samples["speaker_turn"])
            == len(samples["parsed_dialogue"])
            == len(samples["dialogue_sizes"])
            == batch_size
        ):
            raise ValueError("clean model batch dimensions disagree")

        contexts: list[str] = []
        seeker_indices: list[list[int]] = []
        strategy_indices: list[list[tuple[int, int]]] = []
        for dialogue, strategies, speakers, size in zip(
            samples["dialogue_history"],
            samples["strategy_history"],
            samples["speaker_turn"],
            samples["dialogue_sizes"],
        ):
            utterances = dialogue.split("</s>")
            if not (len(utterances) == len(strategies) == len(speakers) == size):
                raise ValueError("clean model sample node dimensions disagree")
            contexts.append(author_context_string(dialogue, speakers))
            sample_seekers: list[int] = []
            sample_strategies: list[tuple[int, int]] = []
            for node_index, (speaker, strategy) in enumerate(
                zip(speakers, strategies)
            ):
                if speaker == "seeker":
                    sample_seekers.append(node_index)
                else:
                    # Preserve upstream: <START>/None and -1 become strategy 0.
                    sample_strategies.append(
                        (node_index, strategy if strategy != -1 else 0)
                    )
            seeker_indices.append(sample_seekers)
            strategy_indices.append(sample_strategies)

        context_embeddings = self._encode_contexts(contexts)
        erc_input = samples["erc_probabilities"].to(
            device=self.device, dtype=torch.float32
        )
        effective_temperature = self.t * self.scalar
        if (
            not torch.isfinite(effective_temperature).item()
            or abs(effective_temperature.detach().item()) < 1e-12
        ):
            raise ValueError("ERC temperature became zero or non-finite")
        # The upstream ERC cache already contains softmax probabilities. The
        # published graph model deliberately applies this second softmax.
        erc_probabilities = self.softmax(erc_input / effective_temperature)
        if self.config_contract.erc_mixed:
            erc_embeddings = erc_probabilities @ self.erc_prototypes
        else:  # frozen config keeps this branch unreachable
            erc_embeddings = self.erc_prototypes[
                torch.argmax(erc_probabilities, dim=-1), :
            ]

        graph_embeddings: list[torch.Tensor] = []
        all_edges: list[list[int]] = []
        all_edge_types: list[int] = []
        dummy_indices: list[int] = []
        graph_offset = 0
        erc_offset = 0
        graph_receipts: list[dict[str, Any]] = []
        for sample_index, size in enumerate(samples["dialogue_sizes"]):
            rows: list[torch.Tensor] = [self.dummy_embedding]
            seeker_set = set(seeker_indices[sample_index])
            strategy_by_node = dict(strategy_indices[sample_index])
            for node_index in range(size):
                if node_index in seeker_set:
                    rows.append(erc_embeddings[erc_offset + node_index])
                else:
                    strategy_tensor = torch.tensor(
                        strategy_by_node[node_index],
                        dtype=torch.long,
                        device=self.device,
                    )
                    rows.append(self.strategy_embedding(strategy_tensor))
            node_embeddings = torch.stack(rows, dim=0)
            graph_embeddings.append(node_embeddings)
            dummy_indices.append(graph_offset)

            local_edges: list[list[int]] = []
            local_types: list[int] = []
            for head, tail, relation in samples["parsed_dialogue"][sample_index]:
                if head != 0:
                    local_edges.append([head, tail])
                    local_types.append(relation)
            for node in range(1, size + 1):
                local_edges.append([node, 0])
                # Upstream calls seeker->dummy Inter=18, all else Self=17.
                local_types.append(18 if node - 1 in seeker_set else 17)
            for head, tail in local_edges:
                all_edges.append([head + graph_offset, tail + graph_offset])
            all_edge_types.extend(local_types)
            graph_receipts.append(
                {"edges": local_edges, "edge_types": local_types}
            )
            graph_offset += size + 1
            erc_offset += size

        values = torch.cat(graph_embeddings, dim=0)
        edge_index = torch.tensor(
            all_edges, dtype=torch.long, device=self.device
        ).t().contiguous()
        edge_type = torch.tensor(
            all_edge_types, dtype=torch.long, device=self.device
        )
        values, attention_1 = self.conv1(values, edge_index, edge_type)
        values, attention_2 = self.conv2(values, edge_index, edge_type)
        values, attention_3 = self.conv3(values, edge_index, edge_type)
        graph_summary = values[
            torch.tensor(dummy_indices, dtype=torch.long, device=self.device)
        ]
        logits = self.classifier(
            torch.cat((graph_summary, context_embeddings), dim=-1)
        )
        return {
            "logits": logits,
            "graphs": graph_receipts,
            "attention_weights": [attention_1, attention_2, attention_3],
            "erc_probabilities_after_author_softmax": erc_probabilities,
        }


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_model_tree(root: Path) -> tuple[str, list[dict[str, Any]]]:
    if root.is_symlink():
        raise ValueError("base model directory must not be a symlink")
    if not root.is_dir():
        raise ValueError(f"base model directory does not exist: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"base model tree contains a symlink: {path}")
        relative = path.relative_to(root)
        if any(part in _MODEL_TREE_IGNORED_PARTS for part in relative.parts):
            continue
        if path.name in _MODEL_TREE_IGNORED_NAMES or not path.is_file():
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not entries:
        raise ValueError("base model tree contains no files")
    return _canonical_json_sha256(entries), entries


def _validate_base_model_tree(
    base_model_path: Path, expected_base_tree_sha256: str
) -> tuple[str, list[dict[str, Any]]]:
    path = Path(base_model_path)
    if not path.is_absolute():
        raise ValueError("base model path must be an absolute local directory")
    if _HEX64.fullmatch(expected_base_tree_sha256) is None:
        raise ValueError("expected base tree SHA-256 is invalid")
    commitment, entries = _hash_model_tree(path)
    if commitment != expected_base_tree_sha256:
        raise ValueError(
            "base model tree commitment mismatch: "
            f"expected={expected_base_tree_sha256} actual={commitment}"
        )
    names = {entry["path"] for entry in entries}
    if "config.json" not in names or "model.safetensors" not in names:
        raise ValueError("base model tree lacks required config or safetensors")
    if not {"tokenizer.json", "vocab.json"}.intersection(names):
        raise ValueError("base model tree lacks tokenizer vocabulary")
    return commitment, entries


def load_clean_emodynamix_model(
    *,
    base_model_path: Path,
    expected_base_tree_sha256: str = EXPECTED_BASE_TREE_SHA256,
    seed: int = 42,
) -> tuple[CleanEmoDynamiXPolicy, dict[str, Any]]:
    """Initialize only from a committed local RoBERTa base tree."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    path = Path(base_model_path)
    commitment, entries = _validate_base_model_tree(
        path, expected_base_tree_sha256
    )
    torch.manual_seed(seed)
    tokenizer = RobertaTokenizer.from_pretrained(
        str(path), local_files_only=True
    )
    roberta = RobertaModel.from_pretrained(
        str(path), local_files_only=True
    )
    config = CleanEmoDynamiXConfig()
    model = CleanEmoDynamiXPolicy(roberta, tokenizer, config)
    # Catch ordinary source mutation while Transformers reopened the files.
    post_commitment, _post_entries = _hash_model_tree(path)
    if post_commitment != commitment:
        raise ValueError("base model tree changed while it was being loaded")
    return model, {
        "model_family": "clean_emodynamix_graph_strategy_policy",
        "upstream_commit": config.upstream_commit,
        "architecture": asdict(config),
        "base_model_path": str(path),
        "base_model_tree_sha256": commitment,
        "base_model_files": entries,
        "local_files_only": True,
        "trust_remote_code": False,
        "symlinks_allowed": False,
        "seed": seed,
        "task_checkpoint_sha256": None,
        "initialization": "roberta_base_plus_random_author_graph_components",
    }


def _parse_strategy_history(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("strategy_history is not valid JSON") from error
    if not isinstance(value, list):
        raise ValueError("strategy_history must be a list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("strategy_history must contain integers")
    return value


def collate_clean_model_batch(
    model_inputs: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Collate label-free inputs and validated precomputed features."""

    if len(model_inputs) != len(feature_rows) or not model_inputs:
        raise ValueError("model inputs and feature rows must have equal nonzero length")
    dialogues: list[str] = []
    strategies_batch: list[list[int]] = []
    speakers_batch: list[list[str]] = []
    parsed_batch: list[list[list[int]]] = []
    erc_rows: list[list[float]] = []
    sizes: list[int] = []

    for sample_index, (model_input, feature) in enumerate(
        zip(model_inputs, feature_rows)
    ):
        if set(model_input) != _MODEL_INPUT_FIELDS:
            raise ValueError("clean model input fields mismatch")
        feature_fields = set(feature)
        if not _MINIMAL_FEATURE_FIELDS.issubset(feature_fields) or not feature_fields.issubset(
            _MINIMAL_FEATURE_FIELDS | _OPTIONAL_FEATURE_FIELDS
        ):
            raise ValueError("clean feature fields mismatch")
        dialogue = model_input["dialogue_history"]
        if not isinstance(dialogue, str):
            raise ValueError("dialogue_history must be a string")
        utterances = [value.strip() for value in dialogue.split("</s>")]
        speakers_value = model_input["speaker_turn"]
        if not isinstance(speakers_value, str):
            raise ValueError("speaker_turn must be a string")
        speakers = speakers_value.split()
        strategies = _parse_strategy_history(model_input["strategy_history"])
        node_count = feature["node_count"]
        if (
            isinstance(node_count, bool)
            or not isinstance(node_count, int)
            or not 1 <= node_count <= 5
            or not len(utterances) == len(speakers) == len(strategies) == node_count
        ):
            raise ValueError(f"sample {sample_index} node dimensions disagree")
        for node_index, (utterance, speaker, strategy) in enumerate(
            zip(utterances, speakers, strategies)
        ):
            if speaker == "seeker" and strategy != -1:
                raise ValueError("seeker nodes must use strategy -1")
            if speaker == "supporter" and not 0 <= strategy <= 7:
                raise ValueError("supporter strategy must be between 0 and 7")
            if speaker == "None":
                if node_index != 0 or utterance != "<START>" or strategy != -1:
                    raise ValueError("None speaker is allowed only for initial <START>")
            elif speaker not in {"seeker", "supporter"}:
                raise ValueError("unknown speaker role")

        probabilities = feature["upstream_erc_softmax_output"]
        if not isinstance(probabilities, list) or len(probabilities) != node_count:
            raise ValueError("ERC rows must match input nodes")
        for vector in probabilities:
            if not isinstance(vector, list) or len(vector) != 7:
                raise ValueError("ERC rows must contain seven probabilities")
            numeric: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("ERC probabilities must be numeric")
                number = float(value)
                if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                    raise ValueError("ERC probabilities must be finite and bounded")
                numeric.append(number)
            if not math.isclose(sum(numeric), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("ERC probabilities must sum to one")
            erc_rows.append(numeric)

        edges = feature["parsed_dialogue"]
        if not isinstance(edges, list):
            raise ValueError("parsed_dialogue must be a list")
        normalized_edges: list[list[int]] = []
        for edge in edges:
            if (
                not isinstance(edge, (list, tuple))
                or len(edge) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in edge
                )
            ):
                raise ValueError("SDDP edge must contain three integers")
            head, tail, relation = edge
            if not 0 <= head <= node_count or not 1 <= tail <= node_count:
                raise ValueError("SDDP edge endpoint is out of range")
            if not 0 <= relation <= 16:
                raise ValueError("SDDP relation must be between 0 and 16")
            normalized_edges.append([head, tail, relation])
        if normalized_edges != sorted(normalized_edges):
            raise ValueError("SDDP edges must be sorted")
        if len(normalized_edges) != len({tuple(edge) for edge in normalized_edges}):
            raise ValueError("SDDP edges must be unique")

        dialogues.append(dialogue)
        strategies_batch.append(strategies)
        speakers_batch.append(speakers)
        parsed_batch.append(normalized_edges)
        sizes.append(node_count)

    return {
        "dialogue_history": dialogues,
        "strategy_history": strategies_batch,
        "speaker_turn": speakers_batch,
        "parsed_dialogue": parsed_batch,
        "erc_probabilities": torch.tensor(
            erc_rows, dtype=torch.float32, device=device
        ),
        "dialogue_sizes": sizes,
    }
