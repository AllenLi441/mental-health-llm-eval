import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = load_module("esconv_corrected", ROOT / "scripts/eval_esconv_joint_corrected.py")
COMPARE = load_module("esconv_compare", ROOT / "scripts/compare_esconv_modern_legacy.py")


def test_checkpoint_token_order_is_exact():
    assert EVAL.EXPECTED_TOKEN_IDS == {
        "[Questions]": 54944,
        "[Reflection of feelings]": 54945,
        "[Information]": 54946,
        "[Restatement or Paraphrasing]": 54947,
        "[Other]": 54948,
        "[Self-disclosure]": 54949,
        "[Affirmation and Reassurance]": 54950,
        "[Providing Suggestions]": 54951,
        "[CLS]": 54952,
    }
    assert "[Question]" not in EVAL.TOKENS_IN_CHECKPOINT_ORDER
    assert "[Others]" not in EVAL.TOKENS_IN_CHECKPOINT_ORDER


def test_gold_is_parsed_from_target_start():
    assert EVAL.parse_target_segment(
        " 1.0 1 4 [Providing Suggestions] You could try writing it down."
    ) == "Providing Suggestions"


def test_target_rejects_nonleading_strategy():
    try:
        EVAL.parse_target_segment("1.0 1 4 Hello [Questions]")
    except ValueError as error:
        assert "begin" in str(error)
    else:
        raise AssertionError("non-leading strategy must be rejected")


def test_metrics_include_invalid_and_confusion_matrix():
    records = [
        {"gold": "Questions", "prediction": "Questions", "correct": True, "invalid": False},
        {"gold": "Other", "prediction": "Questions", "correct": False, "invalid": False},
        {"gold": None, "prediction": None, "correct": False, "invalid": True},
    ]
    metrics = EVAL.compute_metrics(records)
    assert metrics["accuracy"] == 1 / 3
    assert metrics["invalid_rate"] == 1 / 3
    assert metrics["correct"] == 1
    assert metrics["total"] == 3
    assert metrics["confusion_matrix"]["rows_gold_columns_predicted"][0][0] == 1


def test_evaluator_uses_index_select_not_bad_slice():
    source = (ROOT / "scripts/eval_esconv_joint_corrected.py").read_text()
    assert ".index_select(" in source
    assert "54945:54953" not in source
    assert "54_945:54_953" not in source


def test_modern_legacy_comparison():
    modern = {
        0: {"gold": "Questions", "prediction": "Questions", "correct": True},
        1: {"gold": "Other", "prediction": "Questions", "correct": False},
    }
    legacy = {
        0: {"gold": "Questions", "prediction": "Questions", "correct": True},
        1: {"gold": "Other", "prediction": "Other", "correct": True},
    }
    result = COMPARE.compare(modern, legacy)
    assert result["prediction_agreement"] == 0.5
    assert result["modern_accuracy"] == 0.5
    assert result["legacy_accuracy"] == 1.0
    assert result["accuracy_difference_percentage_points"] == -50.0
