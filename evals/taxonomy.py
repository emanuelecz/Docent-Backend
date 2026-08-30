"""Issue taxonomy: maps pydantic's GitHub labels onto triage categories and routes.

Ground truth for classification comes from labels applied by pydantic maintainers
(ingested into ClosedIssue.tags / OpenIssue.tags by ingestion.issues._labels), so
the eval never depends on our own judgement of what an issue "really" is.

Most labels in the repo describe process state (awaiting author response, ready for
review, relnotes-*) rather than issue kind. Those are listed in PROCESS_LABELS and
ignored when deriving a category.
"""

from enum import Enum


class Category(str, Enum):
    SECURITY = "security"
    DUPLICATE = "duplicate"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    CHANGE = "change"
    DOCUMENTATION = "documentation"
    QUESTION = "question"
    UNKNOWN = "unknown"


class Route(str, Enum):
    ANSWER = "answer"
    PRIOR_ART = "prior_art"
    ESCALATE = "escalate"


LABEL_TO_CATEGORY = {
    "security": Category.SECURITY,
    "duplicate": Category.DUPLICATE,
    "bug v1": Category.BUG,
    "bug v2": Category.BUG,
    "feature request": Category.FEATURE_REQUEST,
    "change": Category.CHANGE,
    "refactor": Category.CHANGE,
    "documentation": Category.DOCUMENTATION,
    "question": Category.QUESTION,
}

# Labels that describe workflow state or PR plumbing, not the kind of issue.
PROCESS_LABELS = {
    "awaiting author response",
    "awaiting author revision",
    "dependencies",
    "deferred",
    "feedback wanted",
    "full build",
    "github_actions",
    "meta",
    "needs-blogpost-entry",
    "non-breaking-change",
    "pending",
    "python:uv",
    "ready for review",
    "rust",
}

# An issue can carry several category-bearing labels at once ("duplicate" + "bug V2").
# Earlier entries win. The order is deliberately safety-first: anything that must not
# be auto-answered resolves before anything that may be.
#
# DUPLICATE outranks BUG on purpose: a maintainer marking an issue duplicate is a
# settled triage outcome, and "this was already handled in #N" is both useful and
# cheap to be wrong about. Reorder here if you disagree - the eval will show the cost.
CATEGORY_PRIORITY = [
    Category.SECURITY,
    Category.DUPLICATE,
    Category.BUG,
    Category.FEATURE_REQUEST,
    Category.CHANGE,
    Category.DOCUMENTATION,
    Category.QUESTION,
]

ROUTES = {
    Category.QUESTION: Route.ANSWER,
    Category.DUPLICATE: Route.PRIOR_ART,
    Category.FEATURE_REQUEST: Route.PRIOR_ART,
    Category.CHANGE: Route.PRIOR_ART,
    Category.BUG: Route.ESCALATE,
    Category.DOCUMENTATION: Route.ESCALATE,
    Category.SECURITY: Route.ESCALATE,
    Category.UNKNOWN: Route.ESCALATE,
}

# Categories where posting a wrong answer is materially worse than escalating.
# Used by metrics.unsafe_answer_rate.
MUST_NOT_ANSWER = {Category.BUG, Category.SECURITY, Category.UNKNOWN}


def normalize(label: str) -> str:
    return label.strip().lower()


def category_from_labels(labels) -> Category:
    """Derive the gold category from an issue's GitHub labels.

    Returns UNKNOWN when no label carries a category, which is the common case for
    freshly-opened issues - maintainers add labels later. UNKNOWN is a real outcome,
    not a parse failure: it routes to escalation.
    """
    found = {LABEL_TO_CATEGORY[n] for n in map(normalize, labels or []) if n in LABEL_TO_CATEGORY}
    for category in CATEGORY_PRIORITY:
        if category in found:
            return category
    return Category.UNKNOWN


def route_for(category: Category) -> Route:
    return ROUTES.get(category, Route.ESCALATE)


def unmapped_labels(labels) -> set:
    """Labels that are neither category-bearing nor known process labels.

    Surfaced by evals/label_distribution.py so the taxonomy can be extended from real
    data instead of guesswork - pydantic has ~56 labels and this file covers the ones
    that carry triage meaning.
    """
    return {
        n for n in map(normalize, labels or [])
        if n and n not in LABEL_TO_CATEGORY and n not in PROCESS_LABELS
        and not n.startswith(("relnotes-", "backport-"))
    }
