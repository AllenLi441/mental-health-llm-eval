import json
import unittest

from open_response_eval.core import extract_api_response, parse_strategy_response


class ApiResponseTerminationTests(unittest.TestCase):
    def test_chat_preserves_stop_finish_reason(self):
        result = extract_api_response(
            "chat",
            {
                "id": "chat-stop",
                "model": "model-a",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Complete answer"},
                    }
                ],
                "usage": {"total_tokens": 12},
            },
        )

        self.assertEqual(result["text"], "Complete answer")
        self.assertEqual(result["finish_reason"], "stop")

    def test_chat_preserves_length_finish_reason(self):
        result = extract_api_response(
            "chat",
            {
                "id": "chat-length",
                "model": "model-a",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "Truncated answer"},
                    }
                ],
            },
        )

        self.assertEqual(result["text"], "Truncated answer")
        self.assertEqual(result["finish_reason"], "length")

    def test_responses_preserves_completed_status(self):
        result = extract_api_response(
            "responses",
            {
                "id": "response-complete",
                "model": "model-b",
                "status": "completed",
                "incomplete_details": None,
                "output_text": "Complete answer",
            },
        )

        self.assertEqual(result["text"], "Complete answer")
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["incomplete_details"])

    def test_responses_preserves_incomplete_status_and_details(self):
        details = {"reason": "max_output_tokens"}
        result = extract_api_response(
            "responses",
            {
                "id": "response-incomplete",
                "model": "model-b",
                "status": "incomplete",
                "incomplete_details": details,
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Partial answer"}
                        ],
                    }
                ],
            },
        )

        self.assertEqual(result["text"], "Partial answer")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_details"], details)


class StrictStrategyResponseTests(unittest.TestCase):
    def test_accepts_exact_strategy_label_in_plain_json(self):
        result = parse_strategy_response(
            json.dumps(
                {
                    "strategy": "Reflection of feelings",
                    "response": "That sounds exhausting.",
                }
            )
        )

        self.assertEqual(result.strategy, "Reflection of feelings")
        self.assertEqual(result.response, "That sounds exhausting.")
        self.assertIsNone(result.error)

    def test_rejects_markdown_fenced_json(self):
        result = parse_strategy_response(
            '```json\n{"strategy":"Questions","response":"What happened?"}\n```'
        )

        self.assertIsNone(result.strategy)
        self.assertEqual(result.response, "")
        self.assertEqual(result.error, "invalid_json")

    def test_rejects_lowercase_strategy_label(self):
        result = parse_strategy_response(
            '{"strategy":"questions","response":"What happened?"}'
        )

        self.assertIsNone(result.strategy)
        self.assertEqual(result.response, "What happened?")
        self.assertEqual(result.error, "invalid_strategy")

    def test_rejects_strategy_labels_with_extra_whitespace(self):
        invalid_labels = (
            " Questions",
            "Questions ",
            "Reflection  of feelings",
        )

        for label in invalid_labels:
            with self.subTest(label=label):
                result = parse_strategy_response(
                    json.dumps({"strategy": label, "response": "A response"})
                )

                self.assertIsNone(result.strategy)
                self.assertEqual(result.error, "invalid_strategy")

    def test_empty_response_invalidates_an_exact_strategy(self):
        for response in ("", "   ", "\n\t"):
            with self.subTest(response=repr(response)):
                result = parse_strategy_response(
                    json.dumps({"strategy": "Questions", "response": response})
                )

                self.assertIsNone(result.strategy)
                self.assertEqual(result.response, "")
                self.assertEqual(result.error, "empty_response")

    def test_rejects_extra_schema_keys(self):
        result = parse_strategy_response(
            '{"strategy":"Questions","response":"What happened?","reason":"x"}'
        )

        self.assertIsNone(result.strategy)
        self.assertEqual(result.error, "invalid_schema")


if __name__ == "__main__":
    unittest.main()
