"""Builds labeled eval sets out of issues already in the database.

The gold label is derived from maintainer-applied GitHub labels, never from us.
"""

import random
from collections import defaultdict

from database.db import SessionLocal
from database.models.closed_issue import ClosedIssue
from database.models.open_issue import OpenIssue
from evals.taxonomy import Category, category_from_labels

MODELS = {"closed": ClosedIssue, "open": OpenIssue}


def load_labeled(source="closed", include_unknown=False, limit=None):
    """Load issues with their gold category.

    include_unknown keeps issues whose labels carry no category. They are excluded by
    default when measuring classifier quality (there is nothing to be right about), but
    should be included when measuring end-to-end routing safety, since unlabeled issues
    are exactly what the agent meets in production.
    """
    model = MODELS[source]
    db = SessionLocal()
    try:
        query = db.query(model)
        if limit:
            query = query.limit(limit)
        rows = []
        for issue in query.all():
            category = category_from_labels(issue.tags)
            if category is Category.UNKNOWN and not include_unknown:
                continue
            rows.append({
                "github_number": issue.github_number,
                "title": issue.title,
                "original_question": issue.original_question,
                "url": issue.url,
                "tags": list(issue.tags or []),
                "gold": category,
            })
        return rows
    finally:
        db.close()


def stratified_sample(rows, per_category=15, seed=0):
    """Sample up to per_category items from each gold category.

    Stratified rather than random because the raw distribution is dominated by bugs;
    a random sample would leave single-digit counts on the categories whose routing
    actually differs, and the per-class numbers would be noise.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["gold"]].append(row)

    sample = []
    for category in sorted(buckets, key=lambda c: c.value):
        bucket = sorted(buckets[category], key=lambda r: r["github_number"])
        rng.shuffle(bucket)
        sample.extend(bucket[:per_category])
    return sample
