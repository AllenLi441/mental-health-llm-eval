"""Behavior tests use invented strings, never real counseling examples."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from prepare_a3_candidates import SOURCES, HELDOUT, build, convert, flags


def dialogue(user="invented prompt", answer="invented response"):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


class CandidateTests(unittest.TestCase):
    def test_unknown_roles_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNKNOWN_ROLE"):
            convert("PsyDTCorpus", {"id": 1, "messages": [{"role": "patient", "content": "x"}]}, 1, "a" * 64)

    def test_history_loss_is_zero(self):
        row = {"instruction": None, "history": [["past", "past answer"]], "input": "now", "output": "target"}
        out = convert("MeChat_smile", row, 1, "a" * 64)
        self.assertEqual(out["loss_message_weights"], [0, 0, 0, 1])
        self.assertEqual(out["messages"][-1]["content"], "target")
        self.assertIsNone(out["source_group_id"])
        self.assertFalse(out["product_training_approved"])

    def test_cbt_stages_share_source_group(self):
        rows = [{"patient": "a", "therapist": "b", "profile_id": "p", "stage": s} for s in ["early", "late"]]
        out = [convert("EN_cand_CBT", row, i, "a" * 64) for i, row in enumerate(rows, 1)]
        self.assertEqual(out[0]["source_group_id"], out[1]["source_group_id"])
        self.assertNotEqual(out[0]["record_id"], out[1]["record_id"])

    def test_missing_question_id_is_not_row_lineage(self):
        out = convert("CounselChat", {"questionText": "q", "answerText": "a"}, 33, "a" * 64)
        self.assertIsNone(out["source_group_id"])
        self.assertIsNone(out["split"])

    def test_empty_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "EMPTY_OR_NONSTRING_TEXT"):
            convert("EN_cand_CBT", {"patient": "q", "therapist": " "}, 1, "a" * 64)

    def test_non_alternating_turns_rejected(self):
        with self.assertRaisesRegex(ValueError, "NONALTERNATING_ROLES"):
            convert("PsyDTCorpus", {"messages": [{"role": "assistant", "content": "a"},
                                                   {"role": "assistant", "content": "b"}]}, 1, "a" * 64)

    def test_flag_does_not_become_approval(self):
        self.assertIn("user:email_candidate", flags(dialogue("user@example.test")))
        row = convert("PsyDTCorpus", {"id": 1, "messages": dialogue()}, 1, "a" * 64)
        self.assertEqual(flags(row["messages"]), [])
        self.assertEqual(row["human_review_status"], "NOT_VERIFIED")

    def test_build_determinism_and_holdout_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            for relative, _ in SOURCES.values():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / SOURCES["PsyDTCorpus"][0]).write_text(json.dumps([
                {"id": 1, "messages": dialogue()}, {"id": 1, "messages": dialogue()},
                {"id": 99, "messages": dialogue("different text")},
            ]))
            (root / HELDOUT).write_text(json.dumps([{"id": 99, "messages": dialogue("heldout")}]))
            (root / SOURCES["MeChat_smile"][0]).write_text(json.dumps([
                {"history": [], "input": "smile", "output": "answer", "instruction": None}]))
            with (root / SOURCES["CounselChat"][0]).open("w") as f:
                w = csv.DictWriter(f, fieldnames=["questionID", "questionText", "answerText"])
                w.writeheader(); w.writerow({"questionID": "7", "questionText": "question", "answerText": "answer"})
            (root / SOURCES["EN_cand_CBT"][0]).write_text(json.dumps({"profile_id": "p", "patient": "patient", "therapist": "response"}) + "\n")
            a = build(root, Path(tmp) / "a")
            b = build(root, Path(tmp) / "b")
            self.assertEqual(a, b)
            self.assertEqual(a["raw_pool_records"], 6)
            self.assertEqual(a["normalized_candidate_records"], 4)
            counts = a["sources"]["PsyDTCorpus"]["counts"]
            self.assertEqual(counts["quarantined_heldout_match"], 1)
            self.assertEqual(counts["quarantined_exact_duplicate"], 1)
            self.assertTrue(a["inputs_unchanged"])
            with self.assertRaisesRegex(ValueError, "OUTPUT_MUST_BE_NEW"):
                build(root, Path(tmp) / "a")


if __name__ == "__main__":
    unittest.main()
