import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from open_response_eval.core import CPCDTask
from open_response_eval.runner import (
    EndpointConfig,
    OpenAICompatibleClient,
    load_cpcd_full_history,
    portable_tree_sha256,
    protocol_sha256,
)


class _ChatHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        type(self).requests.append(
            {"path": self.path, "authorization": self.headers.get("Authorization"), "body": body}
        )
        payload = {
            "id": "response-1",
            "model": body["model"],
            "system_fingerprint": "fp-test",
            "choices": [{"message": {"content": "mock answer"}}],
            "usage": {"total_tokens": 9},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format, *_args):
        return


class EndpointTests(unittest.TestCase):
    def test_endpoint_reads_only_its_declared_environment_names(self):
        spec = {
            "id": "qwen",
            "model": "Qwen/Qwen3.6-27B",
            "base_url_env": "QWEN_EVAL_BASE_URL",
            "api_key_env": "QWEN_EVAL_API_KEY",
            "wire_api": "chat",
        }
        environment = {
            "QWEN_EVAL_BASE_URL": "https://qwen.example/v1",
            "QWEN_EVAL_API_KEY": "qwen-key",
            "DEEPSEEK_API_KEY": "must-not-be-used",
        }

        endpoint = EndpointConfig.from_spec(spec, environment)

        self.assertEqual(endpoint.base_url, "https://qwen.example/v1")
        self.assertEqual(endpoint.api_key, "qwen-key")
        self.assertNotIn("qwen-key", repr(endpoint))

    def test_endpoint_fails_closed_when_dedicated_key_is_missing(self):
        spec = {
            "id": "deepseek",
            "model": "deepseek-v4-pro",
            "base_url_env": "EVAL_BASE_URL",
            "api_key_env": "EVAL_API_KEY",
            "wire_api": "chat",
        }

        with self.assertRaisesRegex(ValueError, "EVAL_API_KEY"):
            EndpointConfig.from_spec(spec, {"EVAL_BASE_URL": "https://example.test/v1"})

    def test_client_calls_openai_compatible_chat_endpoint(self):
        _ChatHandler.requests = []
        server = HTTPServer(("127.0.0.1", 0), _ChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = EndpointConfig(
                id="mock",
                model="mock-model",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="secret-test-key",
                wire_api="chat",
                timeout_seconds=3,
                max_retries=0,
            )
            result = OpenAICompatibleClient(endpoint).complete(
                [{"role": "user", "content": "hello"}], max_tokens=50
            )
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()

        self.assertEqual(result["text"], "mock answer")
        self.assertEqual(result["response_model"], "mock-model")
        self.assertEqual(_ChatHandler.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(_ChatHandler.requests[0]["authorization"], "Bearer secret-test-key")
        self.assertEqual(_ChatHandler.requests[0]["body"]["temperature"], 0)


class IntegrityTests(unittest.TestCase):
    def test_portable_tree_hash_includes_relative_paths_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "nested" / "b.txt").write_text("beta", encoding="utf-8")

            first = portable_tree_sha256(root, [root / "a.txt", root / "nested" / "b.txt"])
            (root / "nested" / "b.txt").write_text("changed", encoding="utf-8")
            second = portable_tree_sha256(root, [root / "a.txt", root / "nested" / "b.txt"])

        self.assertNotEqual(first, second)

    def test_protocol_hash_is_stable_across_json_key_order(self):
        self.assertEqual(protocol_sha256({"b": 2, "a": 1}), protocol_sha256({"a": 1, "b": 2}))

    def test_loads_full_history_by_cpcd_source_stem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [{"role": "user", "content": "history"}]
            (root / "case_fullsession.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            task = CPCDTask(
                id="case-mr-1",
                family="mr",
                case_id="case",
                payload={},
                source_file="case.json",
            )

            actual = load_cpcd_full_history(task, root)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
