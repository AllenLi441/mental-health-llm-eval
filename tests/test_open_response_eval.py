import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from open_response_eval.cli import (
    esconv_method_identity,
    run_cpcd_judging,
    run_generation,
    summarize_results,
)

from open_response_eval.core import (
    CPCDTask,
    ESConvExample,
    STRATEGIES,
    build_chat_request,
    build_cpcd_target_messages,
    build_cpcd_judge_messages,
    build_esconv_target_messages,
    extract_api_response,
    load_cpcd_tasks,
    load_esconv_test,
    normalized_quality_score,
    parse_cpcd_judgement,
    parse_strategy_response,
    read_jsonl_index,
    response_overlap_metrics_tokenized,
    score_strategy_predictions,
    summarize_cpcd_judgements,
    stable_blind_order,
    validate_preregistration,
    write_jsonl_record,
)
from open_response_eval.runner import protocol_sha256


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
        self.assertEqual(
            example.context[1]["strategy"], "Reflection of feelings"
        )

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

    def test_empty_response_invalidates_the_strategy_prediction(self):
        parsed = parse_strategy_response(
            '{"strategy":"Questions","response":""}'
        )

        self.assertIsNone(parsed.strategy)
        self.assertEqual(parsed.error, "empty_response")

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

    def test_esconv_prompt_preserves_prior_supporter_strategy_labels(self):
        example = ESConvExample(
            id="esconv-test-1",
            context=[
                {"role": "user", "content": "I feel alone."},
                {
                    "role": "assistant",
                    "content": "That sounds painful.",
                    "strategy": "Reflection of feelings",
                },
                {"role": "user", "content": "I do not know what to do."},
            ],
            gold_strategy="Questions",
            gold_response="Who do you feel safest talking to?",
        )

        messages = build_esconv_target_messages(example, "FROZEN")

        self.assertIn(
            "Supporter: [Reflection of feelings] That sounds painful.",
            messages[1]["content"],
        )


class JudgeTests(unittest.TestCase):
    def test_srg_judge_sees_reference_and_not_model_identity(self):
        task = CPCDTask(
            id="srg-1",
            family="srg",
            case_id="case",
            payload={
                "reference_answer": "A grounded reference answer.",
                "evaluation_focus": {"empathy": "name the mixed feeling"},
                "input_to_model": {
                    "student_profile_summary": "profile",
                    "history_until_previous_session": "history",
                    "current_session_event": "event",
                    "current_session_context": [],
                    "current_student_utterance": "latest",
                },
            },
            source_file="case.json",
        )

        messages = build_cpcd_judge_messages(
            task=task,
            response="Candidate response",
            full_history=None,
            rubric="RUBRIC",
            blind_id="blind-123",
        )
        joined = "\n".join(message["content"] for message in messages)

        self.assertIn("A grounded reference answer.", joined)
        self.assertIn("blind-123", joined)
        self.assertNotIn("deepseek", joined.casefold())
        self.assertNotIn("qwen", joined.casefold())

    def test_parses_valid_srg_judgement_and_rejects_out_of_range(self):
        valid = parse_cpcd_judgement(
            "srg",
            '{"scores":{"empathy":5,"coherence":4,"professionalism":4},'
            '"rationales":{"empathy":"ok","coherence":"ok",'
            '"professionalism":"ok"}}',
        )

        self.assertEqual(valid["scores"]["empathy"], 5)
        self.assertAlmostEqual(valid["average_score"], 13 / 3)
        with self.assertRaisesRegex(ValueError, "range"):
            parse_cpcd_judgement(
                "srg",
                '{"scores":{"empathy":6,"coherence":4,"professionalism":4}}',
            )

    def test_summarizes_cpcd_families_without_calling_them_accuracy(self):
        records = [
            {"family": "srg", "scores": {"empathy": 5, "coherence": 4, "professionalism": 3}},
            {"family": "srg", "scores": {"empathy": 3, "coherence": 4, "professionalism": 5}},
            {
                "family": "mr",
                "scores": {
                    "accuracy": 4,
                    "completeness": 3,
                    "temporal_consistency": 5,
                    "no_hallucination": 4,
                },
            },
        ]

        summary = summarize_cpcd_judgements(records)

        self.assertEqual(summary["families"]["srg"]["n"], 2)
        self.assertEqual(summary["families"]["srg"]["mean_score_0_to_5"], 4.0)
        self.assertEqual(summary["families"]["srg"]["normalized_quality_percent"], 80.0)
        self.assertNotIn("accuracy", summary["overall"])


class StorageAndWireTests(unittest.TestCase):
    def test_extracts_chat_and_responses_payloads(self):
        chat = extract_api_response(
            "chat",
            {
                "model": "model-a",
                "system_fingerprint": "fp-a",
                "choices": [{"message": {"content": "answer-a"}}],
                "usage": {"total_tokens": 10},
            },
        )
        responses = extract_api_response(
            "responses",
            {
                "model": "model-b",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "answer-b"}]}
                ],
                "usage": {"total_tokens": 20},
            },
        )

        self.assertEqual(chat["text"], "answer-a")
        self.assertEqual(chat["fingerprint"], "fp-a")
        self.assertEqual(responses["text"], "answer-b")
        self.assertEqual(responses["response_model"], "model-b")

    def test_jsonl_store_refuses_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            write_jsonl_record(path, {"run_id": "one", "value": 1})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                write_jsonl_record(path, {"run_id": "one", "value": 2})

            index = read_jsonl_index(path)
            self.assertEqual(index["one"]["value"], 1)

    def test_response_overlap_metrics_use_official_percent_scale(self):
        metrics = response_overlap_metrics_tokenized(
            references=[["i", "hear", "you"]],
            hypotheses=[["i", "hear", "you"]],
        )

        self.assertAlmostEqual(metrics["bleu_1"], 100.0)
        self.assertAlmostEqual(metrics["rouge_l"], 100.0)
        self.assertGreater(metrics["distinct_1"], 0.0)

    def test_response_bleu_matches_official_nltk_for_short_and_empty_outputs(self):
        metrics = response_overlap_metrics_tokenized(
            references=[
                ["a", "c", "a", "d", "d", "d", "f", "d", "b", "a"],
                ["f", "b", "e", "a", "c"],
                ["a"],
            ],
            hypotheses=[
                ["a", "d", "d", "e", "a", "f", "d"],
                [],
                ["e", "a", "d", "f", "b", "d", "f", "a", "e", "b"],
            ],
        )

        self.assertAlmostEqual(metrics["bleu_1"], 38.88888888888889)
        self.assertAlmostEqual(metrics["bleu_2"], 27.00308624336608)
        self.assertAlmostEqual(metrics["bleu_3"], 17.334031858765865)
        self.assertAlmostEqual(metrics["bleu_4"], 12.137294292683086)


class EvaluationPipelineTests(unittest.TestCase):
    @staticmethod
    def _preregistration(repository: Path) -> dict:
        return {
            "protocol_id": "test-protocol",
            "target_arms": [
                {
                    "id": "arm-a",
                    "model": "secret-model-a",
                    "default_base_url": "https://arm-a.example/v1",
                    "api_key_optional": True,
                    "wire_api": "chat",
                },
                {
                    "id": "arm-b",
                    "model": "secret-model-b",
                    "revision": "fake-revision",
                    "default_base_url": "https://arm-b.example/v1",
                    "api_key_optional": True,
                    "wire_api": "chat",
                },
            ],
            "judge": {
                "id": "independent-judge",
                "model": "judge-model",
                "default_base_url": "https://judge.example/v1",
                "api_key_optional": True,
                "wire_api": "chat",
                "max_output_tokens": 500,
            },
            "cpcd": {
                "include_reference_answer": True,
                "expected_task_counts": {"srg": 1, "mr": 0, "tcr": 0},
            },
            "esconv": {"expected_rows": 1, "target_max_output_tokens": 64},
            "generation": {"invalid_output_retry_count": 1},
            "randomization": {"blind_order_seed": 20260805},
            "prompts": {
                "zh": {"path": "prompts/zh.txt"},
                "en": {"path": "prompts/en.txt"},
            },
            "sources": {
                "cpcd": {
                    "checkout": "official/cpcd",
                    "eval_root": "eval_task_info",
                },
                "esconv": {
                    "checkout": "official/esconv",
                    "test_file": "test.tsv",
                },
            },
        }

    @staticmethod
    def _write_fake_official_assets(repository: Path) -> None:
        eval_root = repository / "official" / "cpcd" / "eval_task_info"
        for directory in ("srg", "memory_recall", "TCR", "full_session"):
            (eval_root / directory).mkdir(parents=True, exist_ok=True)
        (eval_root / "srg" / "rubric.md").write_text("SRG RUBRIC", encoding="utf-8")
        (eval_root / "memory_recall" / "rubric.md").write_text(
            "MR RUBRIC", encoding="utf-8"
        )
        (eval_root / "TCR" / "rubric.md").write_text("TCR RUBRIC", encoding="utf-8")
        (eval_root / "srg" / "case.json").write_text(
            json.dumps(
                {
                    "task_id": "srg-1",
                    "case_id": "case",
                    "reference_answer": "Reference response",
                    "evaluation_focus": {"empathy": "acknowledge distress"},
                    "input_to_model": {
                        "student_profile_summary": "profile",
                        "history_until_previous_session": "history",
                        "current_session_event": "event",
                        "current_session_context": [],
                        "current_student_utterance": "latest",
                    },
                }
            ),
            encoding="utf-8",
        )
        esconv_root = repository / "official" / "esconv"
        esconv_root.mkdir(parents=True, exist_ok=True)
        (esconv_root / "test.tsv").write_text(
            "1.0 0 0 I feel alone. EOS "
            "1.0 1 1 [Questions] Who do you feel safest talking to?\n",
            encoding="utf-8",
        )
        prompts = repository / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "zh.txt").write_text("FROZEN ZH", encoding="utf-8")
        (prompts / "en.txt").write_text("FROZEN EN", encoding="utf-8")

    @staticmethod
    def _write_generation(
        path: Path,
        preregistration: dict,
        arm_id: str,
        benchmark: str,
        response: str,
    ) -> None:
        task_id = "srg-1" if benchmark == "cpcd" else "esconv-test-000001"
        repository = path.parent
        if benchmark == "cpcd":
            task = load_cpcd_tasks(
                repository / "official" / "cpcd" / "eval_task_info"
            )[0]
            messages = build_cpcd_target_messages(
                task,
                full_history=None,
                jingshi_system_prefix="FROZEN ZH",
            )
        else:
            example = load_esconv_test(repository / "official" / "esconv" / "test.tsv")[0]
            messages = build_esconv_target_messages(
                example,
                jingshi_system_prefix="FROZEN EN",
            )
        messages_sha256 = hashlib.sha256(
            json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        record = {
            "run_id": f"test-protocol:{benchmark}:{arm_id}:{task_id}",
            "protocol_id": "test-protocol",
            "protocol_sha256": protocol_sha256(preregistration),
            "benchmark": benchmark,
            "arm_id": arm_id,
            "task_id": task_id,
            "response": response,
            "messages_sha256": messages_sha256,
            "endpoint_id": arm_id,
            "requested_model": (
                "secret-model-a" if arm_id == "arm-a" else "secret-model-b"
            ),
            "response_model": (
                "secret-model-a" if arm_id == "arm-a" else "secret-model-b"
            ),
        }
        if benchmark == "cpcd":
            record["family"] = "srg"
        else:
            record.update(
                {
                    **esconv_method_identity(),
                    "gold_strategy": "Questions",
                    "predicted_strategy": "Questions",
                    "format_error": None,
                    "format_attempts": 1,
                    "raw_response": json.dumps(
                        {"strategy": "Questions", "response": response}
                    ),
                }
            )
        write_jsonl_record(path, record)

    def test_cpcd_judging_is_blinded_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._write_fake_official_assets(repository)
            preregistration = self._preregistration(repository)
            generations = []
            for arm_id in ("arm-a", "arm-b"):
                path = repository / f"{arm_id}.jsonl"
                self._write_generation(
                    path,
                    preregistration,
                    arm_id,
                    "cpcd",
                    "Candidate response alpha" if arm_id == "arm-a" else "Candidate response beta",
                )
                generations.append(path)
            output = repository / "judgements.jsonl"
            client = Mock()
            client.complete.side_effect = [
                {
                    "text": '{"scores":{"empathy":5,"coherence":4,'
                    '"professionalism":4},"rationales":{}}',
                    "requested_model": "judge-model",
                    "response_model": "judge-model",
                    "endpoint_id": "independent-judge",
                },
                {
                    "text": '{"scores":{"empathy":4,"coherence":4,'
                    '"professionalism":3},"rationales":{}}',
                    "requested_model": "judge-model",
                    "response_model": "judge-model",
                    "endpoint_id": "independent-judge",
                },
            ]
            with (
                patch("open_response_eval.cli.validate_protocol_assets"),
                patch("open_response_eval.cli.OpenAICompatibleClient", return_value=client),
                patch.dict(os.environ, {}, clear=False),
            ):
                first = run_cpcd_judging(
                    preregistration,
                    repository,
                    generations,
                    output,
                    env_file=None,
                    concurrency=1,
                )
                second = run_cpcd_judging(
                    preregistration,
                    repository,
                    generations,
                    output,
                    env_file=None,
                    concurrency=1,
                )
                tampered = [
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                ]
                tampered[0]["messages_sha256"] = "0" * 64
                output.write_text(
                    "\n".join(json.dumps(record) for record in tampered) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "judge prompt mismatch"):
                    run_cpcd_judging(
                        preregistration,
                        repository,
                        generations,
                        output,
                        env_file=None,
                        concurrency=1,
                    )

            records = list(read_jsonl_index(output).values())
            joined_prompts = "\n".join(
                message["content"]
                for call in client.complete.call_args_list
                for message in call.args[0]
            ).casefold()

        self.assertEqual(first["generated_now"], 2)
        self.assertEqual(second["generated_now"], 0)
        self.assertEqual(client.complete.call_count, 2)
        self.assertEqual({record["blind_id"] for record in records}, {"candidate-A", "candidate-B"})
        self.assertEqual({record["arm_id"] for record in records}, {"arm-a", "arm-b"})
        self.assertNotIn("arm-a", joined_prompts)
        self.assertNotIn("arm-b", joined_prompts)
        self.assertNotIn("secret-model", joined_prompts)

    def test_esconv_format_retry_persists_a_verifiable_attempt_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._write_fake_official_assets(repository)
            preregistration = self._preregistration(repository)
            output = repository / "esconv.jsonl"
            client = Mock()
            client.complete.side_effect = [
                {
                    "text": "not json",
                    "endpoint_id": "arm-a",
                    "requested_model": "secret-model-a",
                    "response_model": "secret-model-a",
                },
                {
                    "text": '{"strategy":"Questions","response":"Who can help?"}',
                    "endpoint_id": "arm-a",
                    "requested_model": "secret-model-a",
                    "response_model": "secret-model-a",
                },
            ]
            with (
                patch("open_response_eval.cli.validate_protocol_assets"),
                patch("open_response_eval.cli.OpenAICompatibleClient", return_value=client),
            ):
                result = run_generation(
                    preregistration=preregistration,
                    repository_root=repository,
                    arm_id="arm-a",
                    benchmark="esconv",
                    family=None,
                    limit=1,
                    output=output,
                    env_file=None,
                    concurrency=1,
                )
                resumed = run_generation(
                    preregistration=preregistration,
                    repository_root=repository,
                    arm_id="arm-a",
                    benchmark="esconv",
                    family=None,
                    limit=1,
                    output=output,
                    env_file=None,
                    concurrency=1,
                )
            record = next(iter(read_jsonl_index(output).values()))

        self.assertEqual(result["generated_now"], 1)
        self.assertEqual(resumed["generated_now"], 0)
        self.assertEqual(client.complete.call_count, 2)
        self.assertEqual(record["format_attempts"], 2)
        self.assertEqual(len(record["format_attempt_trace"]), 2)
        self.assertEqual(record["format_attempt_trace"][0]["raw_response"], "not json")
        self.assertNotEqual(
            record["format_attempt_trace"][0]["messages_sha256"],
            record["format_attempt_trace"][1]["messages_sha256"],
        )
        self.assertEqual(
            record["scored_messages_sha256"],
            record["format_attempt_trace"][1]["messages_sha256"],
        )

    def test_cpcd_judging_rejects_mismatched_arm_task_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._write_fake_official_assets(repository)
            preregistration = self._preregistration(repository)
            first = repository / "arm-a.jsonl"
            second = repository / "arm-b.jsonl"
            self._write_generation(first, preregistration, "arm-a", "cpcd", "A")
            second.touch()

            with patch("open_response_eval.cli.validate_protocol_assets"):
                with self.assertRaisesRegex(ValueError, "same CPCD task set"):
                    run_cpcd_judging(
                        preregistration,
                        repository,
                        [first, second],
                        repository / "judgements.jsonl",
                        env_file=None,
                        concurrency=1,
                    )

    def test_summarize_reports_cpcd_quality_and_esconv_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._write_fake_official_assets(repository)
            preregistration = self._preregistration(repository)
            cpcd = repository / "judgements.jsonl"
            cpcd_task = load_cpcd_tasks(
                repository / "official" / "cpcd" / "eval_task_info"
            )[0]
            cpcd_target_messages = build_cpcd_target_messages(
                cpcd_task,
                full_history=None,
                jingshi_system_prefix="FROZEN ZH",
            )
            cpcd_target_hash = hashlib.sha256(
                json.dumps(
                    cpcd_target_messages,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            blind_order = stable_blind_order(
                "srg-1", ["arm-a", "arm-b"], seed=20260805
            )
            candidate_responses = {
                "arm-a": "Candidate response alpha",
                "arm-b": "Candidate response beta",
            }
            cpcd_generation_paths = []
            for arm_id, response in candidate_responses.items():
                generation_path = repository / f"{arm_id}.cpcd.jsonl"
                self._write_generation(
                    generation_path,
                    preregistration,
                    arm_id,
                    "cpcd",
                    response,
                )
                cpcd_generation_paths.append(generation_path)
            for index, arm_id in enumerate(blind_order):
                blind_id = f"candidate-{chr(ord('A') + index)}"
                judge_messages = build_cpcd_judge_messages(
                    task=cpcd_task,
                    response=candidate_responses[arm_id],
                    full_history=None,
                    rubric="SRG RUBRIC",
                    blind_id=blind_id,
                )
                judge_messages_hash = hashlib.sha256(
                    json.dumps(
                        judge_messages,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                write_jsonl_record(
                    cpcd,
                    {
                        "run_id": f"judge:srg-1:{arm_id}",
                        "protocol_id": "test-protocol",
                        "protocol_sha256": protocol_sha256(preregistration),
                        "benchmark": "cpcd",
                        "record_type": "judgement",
                        "arm_id": arm_id,
                        "blind_id": blind_id,
                        "task_id": "srg-1",
                        "family": "srg",
                        "candidate_generation_run_id": f"test-protocol:cpcd:{arm_id}:srg-1",
                        "candidate_response_sha256": hashlib.sha256(
                            candidate_responses[arm_id].encode("utf-8")
                        ).hexdigest(),
                        "candidate_endpoint_id": arm_id,
                        "candidate_requested_model": (
                            "secret-model-a" if arm_id == "arm-a" else "secret-model-b"
                        ),
                        "candidate_response_model": (
                            "secret-model-a" if arm_id == "arm-a" else "secret-model-b"
                        ),
                        "candidate_messages_sha256": cpcd_target_hash,
                        "messages_sha256": judge_messages_hash,
                        "endpoint_id": "independent-judge",
                        "requested_model": "judge-model",
                        "response_model": "judge-model",
                        "judge_raw_response": (
                            '{"scores":{"empathy":5,"coherence":4,'
                            '"professionalism":3},"rationales":{}}'
                        ),
                        "scores": {
                            "empathy": 5,
                            "coherence": 4,
                            "professionalism": 3,
                        },
                    },
                )
            esconv_paths = []
            for arm_id in ("arm-a", "arm-b"):
                esconv = repository / f"{arm_id}.esconv.jsonl"
                self._write_generation(
                    esconv,
                    preregistration,
                    arm_id,
                    "esconv",
                    "Who do you feel safest talking to?",
                )
                esconv_paths.append(esconv)

            summary = summarize_results(
                preregistration,
                repository,
                cpcd_judgement_paths=[cpcd],
                esconv_generation_paths=esconv_paths,
                cpcd_generation_paths=cpcd_generation_paths,
            )

        cpcd_arm = summary["cpcd"]["arms"]["arm-a"]
        esconv_arm = summary["esconv"]["arms"]["arm-a"]
        self.assertEqual(summary["cpcd"]["score_type"], "quality_not_accuracy")
        self.assertEqual(cpcd_arm["families"]["srg"]["mean_score_0_to_5"], 4.0)
        self.assertEqual(cpcd_arm["families"]["srg"]["normalized_quality_percent"], 80.0)
        self.assertTrue(cpcd_arm["coverage"]["complete"])
        self.assertEqual(esconv_arm["strategy"]["accuracy"], 1.0)
        self.assertEqual(esconv_arm["strategy"]["invalid_rate"], 0.0)
        self.assertAlmostEqual(esconv_arm["response"]["bleu_2"], 100.0)
        self.assertTrue(esconv_arm["coverage"]["complete"])
        self.assertTrue(summary["cpcd"]["coverage_complete"])
        self.assertTrue(summary["esconv"]["coverage_complete"])
        self.assertFalse(summary["cpcd"]["formal_claim_ready"])
        self.assertFalse(summary["esconv"]["formal_claim_ready"])

    def test_summarize_rejects_unpaired_arm_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._write_fake_official_assets(repository)
            preregistration = self._preregistration(repository)
            cpcd = repository / "judgements.jsonl"
            cpcd_target_messages = build_cpcd_target_messages(
                load_cpcd_tasks(
                    repository / "official" / "cpcd" / "eval_task_info"
                )[0],
                full_history=None,
                jingshi_system_prefix="FROZEN ZH",
            )
            cpcd_target_hash = hashlib.sha256(
                json.dumps(
                    cpcd_target_messages,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            write_jsonl_record(
                cpcd,
                {
                    "run_id": "judge:srg-1:arm-a",
                    "protocol_id": "test-protocol",
                    "protocol_sha256": protocol_sha256(preregistration),
                    "benchmark": "cpcd",
                    "record_type": "judgement",
                    "arm_id": "arm-a",
                    "blind_id": "candidate-A",
                    "task_id": "srg-1",
                    "family": "srg",
                    "candidate_generation_run_id": "test-protocol:cpcd:arm-a:srg-1",
                    "candidate_response_sha256": "a" * 64,
                    "candidate_endpoint_id": "arm-a",
                    "candidate_requested_model": "secret-model-a",
                    "candidate_response_model": "secret-model-a",
                    "candidate_messages_sha256": cpcd_target_hash,
                    "messages_sha256": "b" * 64,
                    "endpoint_id": "independent-judge",
                    "requested_model": "judge-model",
                    "response_model": "judge-model",
                    "judge_raw_response": (
                        '{"scores":{"empathy":5,"coherence":4,'
                        '"professionalism":3},"rationales":{}}'
                    ),
                    "scores": {"empathy": 5, "coherence": 4, "professionalism": 3},
                },
            )
            esconv = repository / "arm-a.esconv.jsonl"
            self._write_generation(
                esconv,
                preregistration,
                "arm-a",
                "esconv",
                "Who do you feel safest talking to?",
            )
            cpcd_generation = repository / "arm-a.cpcd.jsonl"
            self._write_generation(
                cpcd_generation,
                preregistration,
                "arm-a",
                "cpcd",
                "Candidate response alpha",
            )

            with self.assertRaisesRegex(ValueError, "both target arms"):
                summarize_results(
                    preregistration,
                    repository,
                    cpcd_judgement_paths=[cpcd],
                    esconv_generation_paths=[esconv],
                    cpcd_generation_paths=[cpcd_generation],
                )


if __name__ == "__main__":
    unittest.main()
