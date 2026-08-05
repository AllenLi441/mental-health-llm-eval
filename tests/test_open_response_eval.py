import json
import unittest
from pathlib import Path

from open_response_eval.core import (
    CPCDTask,
    ESConvExample,
    STRATEGIES,
    build_chat_request,
    build_cpcd_target_messages,
    build_esconv_target_messages,
    load_cpcd_tasks,
    load_esconv_test,
    normalized_quality_score,
    parse_strategy_response,
    score_strategy_predictions,
    stable_blind_order,
    validate_preregistration,
)


REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "tmp" / "official_benchmarks"


class ESConvParsingTests(unittest.TestCase):
    def test_parses_official_turn_level_tsv(self):
        line = (
            "1.0 0 0 I feel stuck. EOS "
            "0.0 1 1 [Reflection of feelings] That sounds exhausting. EOS "
            "1.0 0 2 I cannot see a way forward. EOS "
            "1.0 1 3 [Questions] What feels most blocked right now?"
        )

        example = ESConvExample.from_tsv_line(line, line_number=17)

        self.assertEqual(example.id, "esconv-test-000017")
        self.assertEqual(example.gold_strategy, "Questions")
        self.assertEqual(example.gold_response, "What feels most blocked right now?")
        self.assertEqual(
            [message["role"] for message in example.context],
            ["user", "assistant", "user"],
        )
        self.assertNotIn("[Reflection of feelings]", example.context[1]["content"])

    def test_rejects_a_row_whose_target_is_not_supporter(self):
        with self.assertRaisesRegex(ValueError, "supporter"):
            ESConvExample.from_tsv_line(
                "1.0 1 0 [Questions] How are you? EOS 1.0 0 1 Bad.",
                line_number=1,
            )

    def test_loads_pinned_official_test_rows(self):
        path = (
            OFFICIAL
            / "esconv"
            / "codes"
            / "dataset"
            / "testWithStrategy_short.tsv"
        )
        if not path.exists():
            self.skipTest("official ESConv checkout is not present")

        rows = load_esconv_test(path)

        self.assertEqual(len(rows), 2775)
        self.assertTrue(all(row.gold_strategy in STRATEGIES for row in rows))
        self.assertEqual(len({row.id for row in rows}), 2775)


class StrategyOutputTests(unittest.TestCase):
    def test_accepts_strict_json_and_canonicalizes_case_only(self):
        parsed = parse_strategy_response(
            json.dumps(
                {
                    "strategy": "reflection OF feelings",
                    "response": "It sounds as if this has worn you down.",
                }
            )
        )

        self.assertEqual(parsed.strategy, "Reflection of feelings")
        self.assertEqual(parsed.response, "It sounds as if this has worn you down.")
        self.assertIsNone(parsed.error)

    def test_does_not_semantically_map_unknown_labels(self):
        parsed = parse_strategy_response(
            '{"strategy":"Empathy","response":"That sounds hard."}'
        )

        self.assertIsNone(parsed.strategy)
        self.assertEqual(parsed.error, "invalid_strategy")

    def test_strategy_accuracy_counts_invalid_outputs_as_wrong(self):
        summary = score_strategy_predictions(
            [
                ("Questions", "Questions"),
                ("Questions", "Providing Suggestions"),
                ("Providing Suggestions", "Providing Suggestions"),
                ("Other", None),
            ]
        )

        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(summary["invalid"], 1)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertEqual(set(summary["per_class_f1"]), set(STRATEGIES))


class CPCDTests(unittest.TestCase):
    def test_loads_all_three_official_task_families(self):
        path = OFFICIAL / "psy_chronicle" / "eval_task_info"
        if not path.exists():
            self.skipTest("official Psy-Chronicle checkout is not present")

        tasks = load_cpcd_tasks(path)
        counts = {}
        for task in tasks:
            counts[task.family] = counts.get(task.family, 0) + 1

        self.assertEqual(counts, {"srg": 99, "mr": 40, "tcr": 20})
        self.assertEqual(len({task.id for task in tasks}), 159)

    def test_srg_uses_same_frozen_jingshi_prefix_for_both_arms(self):
        task = CPCDTask(
            id="case_srg_1",
            family="srg",
            case_id="case",
            payload={
                "input_to_model": {
                    "student_profile_summary": "A student under academic pressure.",
                    "history_until_previous_session": "The student feared failing.",
                    "current_session_event": "Exam week",
                    "current_session_context": [
                        {"role": "Student", "content": "I froze again."}
                    ],
                    "current_student_utterance": "I think I am hopeless.",
                }
            },
            source_file="case.json",
        )

        messages = build_cpcd_target_messages(
            task,
            full_history=None,
            jingshi_system_prefix="FROZEN JINGSHI CORE",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertTrue(messages[0]["content"].startswith("FROZEN JINGSHI CORE"))
        self.assertIn("current_student_utterance", messages[1]["content"])

    def test_mr_and_tcr_keep_fact_only_official_prompt(self):
        for family in ("mr", "tcr"):
            task = CPCDTask(
                id=f"case_{family}_1",
                family=family,
                case_id="case",
                payload={"question": "What happened first?", "input_to_model": {}},
                source_file="case.json",
            )
            messages = build_cpcd_target_messages(
                task,
                full_history=[{"role": "user", "content": "History"}],
                jingshi_system_prefix="FROZEN JINGSHI CORE",
            )

            self.assertNotIn("FROZEN JINGSHI CORE", messages[0]["content"])
            self.assertIn("History", messages[1]["content"])

    def test_normalized_quality_is_not_named_accuracy(self):
        score = normalized_quality_score(3.75)

        self.assertEqual(score, {"mean_score_0_to_5": 3.75, "normalized_quality_percent": 75.0})


class ProtocolSafetyTests(unittest.TestCase):
    def test_blind_order_is_deterministic_and_balanced_by_seed(self):
        first = stable_blind_order("task-1", ["deepseek", "qwen"], seed=20260805)
        second = stable_blind_order("task-1", ["deepseek", "qwen"], seed=20260805)

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"deepseek", "qwen"})

    def test_rejects_target_model_as_judge(self):
        prereg = {
            "target_arms": [
                {"id": "deepseek", "model": "deepseek-v4-pro"},
                {"id": "qwen", "model": "Qwen/Qwen3.6-27B"},
            ],
            "judge": {"model": "deepseek-v4-pro"},
            "cpcd": {"include_reference_answer": True},
        }

        with self.assertRaisesRegex(ValueError, "independent"):
            validate_preregistration(prereg)

    def test_requires_cpcd_reference_for_paper_aligned_judging(self):
        prereg = {
            "target_arms": [
                {"id": "deepseek", "model": "deepseek-v4-pro"},
                {"id": "qwen", "model": "Qwen/Qwen3.6-27B"},
            ],
            "judge": {"model": "openai/gpt-5.2"},
            "cpcd": {"include_reference_answer": False},
        }

        with self.assertRaisesRegex(ValueError, "reference"):
            validate_preregistration(prereg)

    def test_builds_chat_and_responses_wire_payloads(self):
        messages = [{"role": "user", "content": "hello"}]

        chat = build_chat_request("chat", "model-a", messages, max_tokens=200)
        responses = build_chat_request("responses", "model-b", messages, max_tokens=300)

        self.assertEqual(chat["path"], "/chat/completions")
        self.assertEqual(chat["body"]["temperature"], 0)
        self.assertEqual(chat["body"]["max_tokens"], 200)
        self.assertEqual(responses["path"], "/responses")
        self.assertNotIn("temperature", responses["body"])
        self.assertEqual(responses["body"]["max_output_tokens"], 300)

    def test_esconv_prompt_requires_exact_strategy_and_response_json(self):
        example = ESConvExample(
            id="esconv-test-1",
            context=[{"role": "user", "content": "I feel alone."}],
            gold_strategy="Questions",
            gold_response="Who do you feel safest talking to?",
        )

        messages = build_esconv_target_messages(
            example,
            jingshi_system_prefix="FROZEN JINGSHI CORE",
        )

        self.assertTrue(messages[0]["content"].startswith("FROZEN JINGSHI CORE"))
        self.assertIn(json.dumps(STRATEGIES), messages[0]["content"])
        self.assertIn('"strategy"', messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
