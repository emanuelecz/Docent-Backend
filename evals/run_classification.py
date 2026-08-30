"""Score the intake classifier against maintainer labels.

    python -m evals.label_distribution              # look at the data first
    python -m evals.run_classification --per-category 15

Writes every prediction to evals/results/ so disagreements can be read by hand - the
misclassified rows are the raw material for the next prompt fix, and reading them is
how you build the domain judgement the rubric needs.
"""

import argparse
import json
import os
from datetime import datetime, timezone

from ai.classify_llm import ClassificationError, classify_issue
from evals.dataset import load_labeled, stratified_sample
from evals.metrics import format_report, worst_confusions
from evals.taxonomy import Category, route_for

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run(rows, verbose=False):
    records = []
    for i, row in enumerate(rows, 1):
        try:
            prediction = classify_issue(row["title"], row["original_question"])
        except ClassificationError as exc:
            # A malformed response is a real outcome, not a crash: count it as an
            # abstention so it shows up in the numbers instead of aborting the run.
            prediction = {"category": Category.UNKNOWN, "confidence": None, "reason": str(exc)}

        record = {
            "github_number": row["github_number"],
            "url": row["url"],
            "title": row["title"],
            "tags": row["tags"],
            "gold": row["gold"].value,
            "predicted": prediction["category"].value,
            "confidence": prediction["confidence"],
            "reason": prediction["reason"],
            "correct": row["gold"] is prediction["category"],
            "gold_route": route_for(row["gold"]).value,
            "predicted_route": route_for(prediction["category"]).value,
        }
        records.append(record)

        if verbose:
            mark = "ok " if record["correct"] else "MISS"
            print(f"[{i}/{len(rows)}] {mark} #{record['github_number']} "
                  f"gold={record['gold']} pred={record['predicted']}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["closed", "open"], default="closed")
    parser.add_argument("--per-category", type=int, default=15,
                        help="max issues sampled per gold category")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-unknown", action="store_true",
                        help="also score issues with no category label (routing safety view)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows loaded from the DB before sampling")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    rows = load_labeled(source=args.source, include_unknown=args.include_unknown,
                        limit=args.limit)
    if not rows:
        raise SystemExit(
            "No labeled issues found. Ingest first, and check "
            "`python -m evals.label_distribution` to confirm tags are populated."
        )

    sample = stratified_sample(rows, per_category=args.per_category, seed=args.seed)
    print(f"scoring {len(sample)} issues from {len(rows)} labeled {args.source} issues\n")

    records = run(sample, verbose=not args.quiet)
    pairs = [(Category(r["gold"]), Category(r["predicted"])) for r in records]

    print()
    print(format_report(pairs, title=f"intake classification ({args.source})"))

    confusions = worst_confusions(pairs)
    if confusions:
        print("\n-- most frequent mistakes --")
        for (gold, predicted), n in confusions:
            print(f"  {gold.value:<16} -> {predicted.value:<16} {n:>3}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RESULTS_DIR, f"classification-{args.source}-{stamp}.json")
    with open(path, "w") as fh:
        json.dump({
            "source": args.source,
            "per_category": args.per_category,
            "seed": args.seed,
            "n": len(records),
            "records": records,
        }, fh, indent=2)

    misses = [r for r in records if not r["correct"]]
    print(f"\nwrote {path}")
    print(f"{len(misses)} disagreements - read them, they are the next prompt fix")


if __name__ == "__main__":
    main()
