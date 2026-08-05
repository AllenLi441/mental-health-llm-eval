import unittest
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from open_response_eval.cli import (
    DEFAULT_PREREGISTRATION,
    build_dry_run_summary,
    load_preregistration,
    main,
    validate_protocol_assets,
)


REPO = Path(__file__).resolve().parents[1]


class _GenerationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        payload = {
            "id": "mock-generation",
            "model": body["model"],
            "system_fingerprint": "fp-mock",
            "choices": [{"message": {"content": "A grounded mock response."}}],
            "usage": {"total_tokens": 12},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format, *_args):
        return


class PreregistrationIntegrationTests(unittest.TestCase):
    def test_repository_preregistration_is_frozen_and_self_consistent(self):
        prereg = load_preregistration(DEFAULT_PREREGISTRATION)

        self.assertEqual(prereg["protocol_id"], "jingshi-cpcd-esconv-20260805-v1")
        self.assertEqual(
            [arm["model"] for arm in prereg["target_arms"]],
            ["deepseek-v4-pro", "Qwen/Qwen3.6-27B"],
        )
        self.assertEqual(prereg["judge"]["model"], "openai/gpt-5.2")
        self.assertTrue(prereg["cpcd"]["include_reference_answer"])
        self.assertEqual(prereg["esconv"]["expected_rows"], 2775)

    def test_official_assets_match_frozen_commits_and_hashes(self):
        prereg = load_preregistration(DEFAULT_PREREGISTRATION)
        psy_checkout = REPO / prereg["sources"]["cpcd"]["checkout"]
        esconv_checkout = REPO / prereg["sources"]["esconv"]["checkout"]
        if not psy_checkout.exists() or not esconv_checkout.exists():
            self.skipTest("official benchmark checkouts are not present")

        status = validate_protocol_assets(prereg, REPO)

        self.assertEqual(status["cpcd"]["task_counts"], {"srg": 99, "mr": 40, "tcr": 20})
        self.assertEqual(status["esconv"]["rows"], 2775)
        self.assertTrue(status["prompts"]["zh"]["sha256_ok"])
        self.assertTrue(status["prompts"]["en"]["sha256_ok"])

    def test_dry_run_lists_calls_without_requiring_credentials(self):
        prereg = load_preregistration(DEFAULT_PREREGISTRATION)

        summary = build_dry_run_summary(prereg, REPO)

        self.assertEqual(summary["target_calls_per_arm"]["cpcd"], 159)
        self.assertEqual(summary["target_calls_per_arm"]["esconv"], 2775)
        self.assertEqual(summary["judge_calls"]["cpcd"], 318)
        self.assertFalse(summary["reports_single_accuracy"])

    def test_generate_command_writes_one_resumable_cpcd_record(self):
        cpcd_checkout = REPO / "tmp" / "official_benchmarks" / "psy_chronicle"
        if not cpcd_checkout.exists():
            self.skipTest("official benchmark checkout is not present")
        server = HTTPServer(("127.0.0.1", 0), _GenerationHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "generation.jsonl"
                environment = {
                    "EVAL_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                    "EVAL_API_KEY": "mock-key",
                    "EVAL_WIRE_API": "chat",
                }
                with patch.dict(os.environ, environment, clear=False):
                    exit_code = main(
                        [
                            "generate",
                            "--arm",
                            "jingshi_deepseek_v4_pro",
                            "--benchmark",
                            "cpcd",
                            "--family",
                            "srg",
                            "--limit",
                            "1",
                            "--output",
                            str(output),
                        ]
                    )
                record = json.loads(output.read_text(encoding="utf-8").strip())
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()

        self.assertEqual(exit_code, 0)
        self.assertEqual(record["arm_id"], "jingshi_deepseek_v4_pro")
        self.assertEqual(record["benchmark"], "cpcd")
        self.assertEqual(record["family"], "srg")
        self.assertEqual(record["response"], "A grounded mock response.")
        self.assertEqual(record["response_model"], "deepseek-v4-pro")
        self.assertNotIn("messages", record)


if __name__ == "__main__":
    unittest.main()
