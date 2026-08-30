"""Print the real label and category distribution of the ingested corpus.

Run this before trusting any taxonomy decision - it reports which labels actually
occur, how they map onto categories, what share of issues arrive with no category
label at all, and which labels the taxonomy does not yet cover.

    python -m evals.label_distribution
"""

import argparse
from collections import Counter

from evals.dataset import MODELS
from evals.taxonomy import Category, category_from_labels, route_for, unmapped_labels
from database.db import SessionLocal


def report(source):
    model = MODELS[source]
    db = SessionLocal()
    try:
        rows = db.query(model.github_number, model.tags).all()
    finally:
        db.close()

    labels = Counter()
    categories = Counter()
    routes = Counter()
    unmapped = Counter()
    unlabeled = 0

    for _, tags in rows:
        tags = list(tags or [])
        if not tags:
            unlabeled += 1
        labels.update(t.strip().lower() for t in tags)
        category = category_from_labels(tags)
        categories[category] += 1
        routes[route_for(category)] += 1
        unmapped.update(unmapped_labels(tags))

    total = len(rows) or 1
    print(f"\n===== {source} issues: {len(rows)} =====\n")

    print("-- category (gold, from maintainer labels) --")
    for category, n in categories.most_common():
        print(f"  {category.value:<16} {n:>5}  {n / total:6.1%}")

    print("\n-- resulting route --")
    for route, n in routes.most_common():
        print(f"  {route.value:<16} {n:>5}  {n / total:6.1%}")

    print(f"\n-- coverage --")
    print(f"  no labels at all      {unlabeled:>5}  {unlabeled / total:6.1%}")
    print(f"  no category label     {categories[Category.UNKNOWN]:>5}  "
          f"{categories[Category.UNKNOWN] / total:6.1%}")
    print("  (a high share here on OPEN issues is expected - maintainers label later -")
    print("   and is why the runtime classifier must not read tags)")

    print("\n-- top raw labels --")
    for label, n in labels.most_common(25):
        print(f"  {label:<28} {n:>5}")

    if unmapped:
        print("\n-- labels not covered by evals/taxonomy.py --")
        for label, n in unmapped.most_common(20):
            print(f"  {label:<28} {n:>5}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["closed", "open", "both"], default="both")
    args = parser.parse_args()

    sources = ["closed", "open"] if args.source == "both" else [args.source]
    for source in sources:
        report(source)


if __name__ == "__main__":
    main()
