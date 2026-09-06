import json
import tempfile
import unittest
from pathlib import Path

from materialize_provisional_training import materialize


def candidate(record_id, group="g", flags=None):
    return {
        "record_id": record_id, "source_revision": "r", "source_group_id": group, "source_row_1based": 1,
        "language": "en", "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
        "target_message_index": 1, "content_state": "CANDIDATE_NOT_FROZEN",
        "product_training_approved": False, "human_review_status": "NOT_VERIFIED",
        "review_flags": flags or [], "target_selection": "source_provided_final_target",
        "synthetic_origin": "PUBLIC_QA",
    }


class MaterializeTests(unittest.TestCase):
    def test_split_and_quarantine_are_complete_and_group_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "c", Path(tmp) / "o"; root.mkdir()
            rows = [candidate("ok", "group-ok"), candidate("flag", "group-flag", ["user:acute_risk_cue"]), candidate("nogroup", None)]
            (root / "Example.candidates.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            result = materialize(root, out)
            self.assertEqual(result["counts"], {"candidate_rows_seen": 3, "train": 1, "development": 0, "quarantined": 2})
            with (out / "quarantine_records.jsonl").open() as handle:
                self.assertEqual(sum(1 for _ in handle), 2)
            self.assertEqual(result["group_counts"]["train"] + result["group_counts"]["development"], 1)
            self.assertFalse(result["training_allowed"])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "c", Path(tmp) / "o"; root.mkdir(); out.mkdir()
            (root / "Example.candidates.jsonl").write_text(json.dumps(candidate("ok")) + "\n")
            with self.assertRaisesRegex(ValueError, "new directory"):
                materialize(root, out)


if __name__ == "__main__":
    unittest.main()
