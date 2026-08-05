import unittest
from pathlib import Path

from open_response_eval.cli import (
    DEFAULT_PREREGISTRATION,
    build_dry_run_summary,
    load_preregistration,
    validate_protocol_assets,
)


REPO = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
