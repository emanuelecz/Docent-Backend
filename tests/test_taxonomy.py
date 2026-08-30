from evals.taxonomy import (
    Category,
    Route,
    category_from_labels,
    route_for,
    unmapped_labels,
)


def test_maps_real_pydantic_labels():
    assert category_from_labels(["bug V2"]) is Category.BUG
    assert category_from_labels(["bug V1"]) is Category.BUG
    assert category_from_labels(["feature request"]) is Category.FEATURE_REQUEST
    assert category_from_labels(["change"]) is Category.CHANGE
    assert category_from_labels(["question"]) is Category.QUESTION
    assert category_from_labels(["documentation"]) is Category.DOCUMENTATION
    assert category_from_labels(["security"]) is Category.SECURITY


def test_label_matching_is_case_and_space_insensitive():
    assert category_from_labels(["  Bug V2 "]) is Category.BUG


def test_process_labels_carry_no_category():
    assert category_from_labels(["pending", "awaiting author response"]) is Category.UNKNOWN
    assert category_from_labels(["relnotes-fix", "backport-2.13"]) is Category.UNKNOWN


def test_process_labels_do_not_mask_a_real_category():
    assert category_from_labels(["pending", "question"]) is Category.QUESTION


def test_unlabeled_issue_is_unknown_not_an_error():
    assert category_from_labels([]) is Category.UNKNOWN
    assert category_from_labels(None) is Category.UNKNOWN


def test_priority_resolves_multiple_category_labels():
    # security outranks everything
    assert category_from_labels(["question", "security"]) is Category.SECURITY
    # duplicate outranks bug, by documented choice
    assert category_from_labels(["bug V2", "duplicate"]) is Category.DUPLICATE
    # bug outranks a co-applied question
    assert category_from_labels(["question", "bug V2"]) is Category.BUG


def test_routing_never_answers_the_dangerous_categories():
    for category in (Category.BUG, Category.SECURITY, Category.UNKNOWN):
        assert route_for(category) is Route.ESCALATE


def test_only_questions_are_answered_directly():
    answered = [c for c in Category if route_for(c) is Route.ANSWER]
    assert answered == [Category.QUESTION]


def test_suggestion_categories_get_prior_art_not_an_opinion():
    assert route_for(Category.FEATURE_REQUEST) is Route.PRIOR_ART
    assert route_for(Category.CHANGE) is Route.PRIOR_ART
    assert route_for(Category.DUPLICATE) is Route.PRIOR_ART


def test_every_category_has_a_route():
    for category in Category:
        assert isinstance(route_for(category), Route)


def test_unmapped_labels_ignores_known_and_generated_families():
    assert unmapped_labels(["question", "relnotes-fix", "backport-2.13", "pending"]) == set()
    assert unmapped_labels(["brand-new-label"]) == {"brand-new-label"}
