import unittest

from candidate_training_guard import check_action


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {"content_state": "CANDIDATE_NOT_FROZEN"}

    def test_candidate_allows_only_preparation(self):
        self.assertEqual(check_action(self.candidate, "audit")[0], True)
        self.assertEqual(check_action(self.candidate, "local_rebuild")[0], True)
        self.assertEqual(check_action(self.candidate, "handoff")[0], True)

    def test_candidate_rejects_every_training_or_release_action(self):
        for action in ("train", "remote_sft", "product_sft", "release"):
            allowed, reason = check_action(self.candidate, action)
            self.assertFalse(allowed, action)
            self.assertIn("qualified export", reason)

    def test_unknown_and_state_drift_fail_closed(self):
        self.assertFalse(check_action(self.candidate, "anything")[0])
        self.assertFalse(check_action({"content_state": "FROZEN_EXPORT"}, "audit")[0])


if __name__ == "__main__":
    unittest.main()
