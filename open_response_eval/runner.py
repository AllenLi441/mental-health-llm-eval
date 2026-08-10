from __future__ import annotations

import hashlib
import json
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import (
    CPCDTask,
    build_chat_request,
    extract_api_response,
    read_jsonl_index,
)


def build_ssl_context(cafile: str | None = None) -> ssl.SSLContext:
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


@dataclass(frozen=True)
class EndpointConfig:
    id: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    wire_api: str = "chat"
    timeout_seconds: float = 180.0
    max_retries: int = 3
    request_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(
        cls, spec: Mapping[str, Any], environment: Mapping[str, str]
    ) -> "EndpointConfig":
        endpoint_id = str(spec.get("id") or "").strip()
        model = str(spec.get("model") or "").strip()
        if not endpoint_id or not model:
            raise ValueError("endpoint id and model are required")

        base_url_env = str(spec.get("base_url_env") or "").strip()
        base_url = str(
            (environment.get(base_url_env) if base_url_env else "")
            or spec.get("default_base_url")
            or ""
        ).strip()
        if not base_url:
            label = base_url_env or "base_url"
            raise ValueError(f"missing endpoint base URL; set {label}")

        api_key_env = str(spec.get("api_key_env") or "").strip()
        api_key = str(environment.get(api_key_env, "") if api_key_env else "").strip()
        if not api_key and not bool(spec.get("api_key_optional")):
            label = api_key_env or "dedicated API key"
            raise ValueError(f"missing dedicated credential; set {label}")

        wire_api_env = str(spec.get("wire_api_env") or "").strip()
        wire_api = str(
            (environment.get(wire_api_env) if wire_api_env else "")
            or spec.get("wire_api")
            or "chat"
        ).strip()
        if wire_api not in {"chat", "responses"}:
            raise ValueError(f"unsupported wire API: {wire_api}")
        request_overrides = dict(spec.get("request_overrides") or {})
        reserved = {
            "model",
            "messages",
            "input",
            "temperature",
            "max_tokens",
            "max_output_tokens",
        }
        overlap = reserved.intersection(request_overrides)
        if overlap:
            raise ValueError(f"request_overrides contains reserved keys: {sorted(overlap)}")
        return cls(
            id=endpoint_id,
            model=model,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            wire_api=wire_api,
            timeout_seconds=float(spec.get("timeout_seconds", 180.0)),
            max_retries=int(spec.get("max_retries", 3)),
            request_overrides=request_overrides,
        )


class OpenAICompatibleClient:
    def __init__(self, endpoint: EndpointConfig):
        self.endpoint = endpoint
        self.ssl_context = build_ssl_context()

    def complete(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> dict[str, Any]:
        request_data = build_chat_request(
            self.endpoint.wire_api,
            self.endpoint.model,
            messages,
            max_tokens=max_tokens,
        )
        request_data["body"].update(self.endpoint.request_overrides)
        url = f"{self.endpoint.base_url}{request_data['path']}"
        encoded = json.dumps(request_data["body"]).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"

        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.endpoint.max_retries + 1):
            request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
            try:
                open_kwargs: dict[str, Any] = {"timeout": self.endpoint.timeout_seconds}
                if url.startswith("https://"):
                    open_kwargs["context"] = self.ssl_context
                with urllib.request.urlopen(request, **open_kwargs) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                parsed = extract_api_response(self.endpoint.wire_api, payload)
                return {
                    **parsed,
                    "requested_model": self.endpoint.model,
                    "endpoint_id": self.endpoint.id,
                    "wire_api": self.endpoint.wire_api,
                    "max_output_tokens_requested": max_tokens,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "attempts": attempt + 1,
                }
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"HTTP {error.code}: {body}")
                retryable = error.code == 429 or error.code >= 500
                if not retryable or attempt >= self.endpoint.max_retries:
                    break
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2**attempt
                time.sleep(min(max(delay, 0.1), 30.0))
            except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt >= self.endpoint.max_retries:
                    break
                time.sleep(min(2**attempt, 30.0))
        raise RuntimeError(
            f"endpoint {self.endpoint.id} failed after "
            f"{self.endpoint.max_retries + 1} attempt(s): {last_error}"
        ) from last_error


def protocol_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def portable_tree_sha256(root: Path, files: Sequence[Path]) -> str:
    root = root.resolve()
    hasher = hashlib.sha256()
    ordered = sorted((path.resolve() for path in files), key=lambda path: path.relative_to(root).as_posix())
    for path in ordered:
        try:
            relative = path.relative_to(root).as_posix().encode("utf-8")
        except ValueError as error:
            raise ValueError(f"tree hash file is outside root: {path}") from error
        content = path.read_bytes()
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)
    return hasher.hexdigest()


def load_cpcd_full_history(task: CPCDTask, full_session_dir: Path) -> Any:
    if task.family == "srg":
        return None
    path = full_session_dir / f"{Path(task.source_file).stem}_fullsession.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing CPCD full history for {task.id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class JsonlRunStore:
    def __init__(self, path: Path, id_key: str = "run_id"):
        self.path = path
        self.id_key = id_key
        self.index = read_jsonl_index(
            path, id_key=id_key, recover_truncated_tail=True
        )

    def append(self, record: dict[str, Any]) -> None:
        run_id = str(record.get(self.id_key) or "")
        if not run_id:
            raise ValueError(f"record is missing {self.id_key}")
        if run_id in self.index:
            raise ValueError(f"duplicate {self.id_key} {run_id!r} in {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        with self.path.open("ab+") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if size:
                handle.seek(-1, 2)
                if handle.read(1) not in (b"\n", b"\r"):
                    handle.write(b"\n")
            handle.write(encoded + b"\n")
        self.index[run_id] = record
