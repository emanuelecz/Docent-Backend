import pytest

from evals.metrics import (
    accuracy,
    confusion_matrix,
    escalation_rate,
    macro_f1,
    per_class_metrics,
    route_pairs,
    unsafe_answer_rate,
    worst_confusions,
)
from evals.taxonomy import Category, Route

Q = Category.QUESTION
B = Category.BUG
F = Category.FEATURE_REQUEST


def test_accuracy_and_empty_input():
    assert accuracy([(Q, Q), (B, B), (Q, B)]) == pytest.approx(2 / 3)
    assert accuracy([]) == 0.0


def test_confusion_matrix_counts_pairs():
    matrix = confusion_matrix([(Q, Q), (Q, B), (Q, B)])
    assert matrix[(Q, Q)] == 1
    assert matrix[(Q, B)] == 2


def test_per_class_precision_and_recall():
    # two questions, one caught; one bug predicted question
    pairs = [(Q, Q), (Q, B), (B, Q)]
    metrics = per_class_metrics(pairs)
    assert metrics[Q]["recall"] == pytest.approx(0.5)
    assert metrics[Q]["precision"] == pytest.approx(0.5)
    assert metrics[Q]["support"] == 2
    assert metrics[Q]["predicted"] == 2


def test_per_class_handles_class_never_predicted():
    metrics = per_class_metrics([(Q, B), (Q, B)])
    assert metrics[Q]["precision"] == 0.0
    assert metrics[Q]["recall"] == 0.0
    assert metrics[Q]["f1"] == 0.0


def test_macro_f1_ignores_classes_with_no_gold_support():
    # F is only ever predicted, never gold: it must not drag the macro average
    pairs = [(Q, Q), (B, F)]
    metrics = per_class_metrics(pairs)
    assert metrics[F]["support"] == 0
    assert macro_f1(pairs) == pytest.approx((1.0 + 0.0) / 2)


def test_macro_f1_weights_small_classes_equally():
    # 9 bugs all correct, 1 question wrong: accuracy is high, macro F1 is not
    pairs = [(B, B)] * 9 + [(Q, B)]
    assert accuracy(pairs) == pytest.approx(0.9)
    assert macro_f1(pairs) < 0.6


def test_unsafe_answer_rate_flags_bug_routed_to_answer():
    # a bug predicted as a question would be answered publicly - the failure that matters
    assert unsafe_answer_rate([(B, Q)]) == 1.0
    assert unsafe_answer_rate([(B, B)]) == 0.0


def test_unsafe_answer_rate_ignores_safe_misclassifications():
    # wrong, but both routes escalate, so nothing gets posted
    assert unsafe_answer_rate([(B, Category.DOCUMENTATION)]) == 0.0


def test_unsafe_answer_rate_denominator_is_must_not_answer_only():
    pairs = [(B, Q), (Q, Q), (Q, Q), (Q, Q)]
    assert unsafe_answer_rate(pairs) == 1.0


def test_unsafe_answer_rate_empty_is_zero():
    assert unsafe_answer_rate([]) == 0.0
    assert unsafe_answer_rate([(Q, Q)]) == 0.0


def test_route_pairs_projects_onto_actions():
    assert route_pairs([(B, Q)]) == [(Route.ESCALATE, Route.ANSWER)]


def test_escalation_rate_uses_predictions():
    assert escalation_rate([(Q, B), (Q, Q)]) == pytest.approx(0.5)
    assert escalation_rate([]) == 0.0


def test_worst_confusions_ranks_by_frequency():
    pairs = [(Q, B), (Q, B), (B, Q), (Q, Q)]
    top = worst_confusions(pairs)
    assert top[0] == ((Q, B), 2)
    assert ((Q, Q), 1) not in top
