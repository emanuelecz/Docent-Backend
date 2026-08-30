"""Classification metrics for the intake eval.

Pure functions over (gold, predicted) pairs - no DB, no API, no numpy - so the whole
module is testable offline and cheap to reason about.
"""

from collections import Counter, defaultdict

from evals.taxonomy import Category, MUST_NOT_ANSWER, Route, route_for


def confusion_matrix(pairs):
    """{(gold, predicted): count} for every observed combination."""
    return Counter((gold, predicted) for gold, predicted in pairs)


def accuracy(pairs):
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(1 for gold, predicted in pairs if gold == predicted) / len(pairs)


def per_class_metrics(pairs):
    """Precision / recall / F1 / support for every class present as gold or prediction."""
    pairs = list(pairs)
    tp = Counter()
    predicted_count = Counter()
    gold_count = Counter()

    for gold, predicted in pairs:
        gold_count[gold] += 1
        predicted_count[predicted] += 1
        if gold == predicted:
            tp[gold] += 1

    out = {}
    for label in set(gold_count) | set(predicted_count):
        precision = tp[label] / predicted_count[label] if predicted_count[label] else 0.0
        recall = tp[label] / gold_count[label] if gold_count[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": gold_count[label],
            "predicted": predicted_count[label],
        }
    return out


def macro_f1(pairs):
    """Unweighted mean F1 over classes that actually occur as gold.

    Macro rather than micro on purpose: the category distribution is heavily skewed
    towards bugs, and a micro average would let the classifier score well while being
    useless on the small categories the agent most needs to get right.
    """
    metrics = per_class_metrics(pairs)
    present = [m for label, m in metrics.items() if m["support"] > 0]
    if not present:
        return 0.0
    return sum(m["f1"] for m in present) / len(present)


def unsafe_answer_rate(pairs):
    """Share of must-not-answer issues the classifier would route to ANSWER.

    This is the number that matters. Everything else on this page can look fine while
    the agent still posts a confident reply on a security report or an unlabeled issue.
    Denominator is issues whose *gold* category must not be answered.
    """
    pairs = list(pairs)
    at_risk = [(gold, predicted) for gold, predicted in pairs if gold in MUST_NOT_ANSWER]
    if not at_risk:
        return 0.0
    unsafe = sum(1 for _, predicted in at_risk if route_for(predicted) is Route.ANSWER)
    return unsafe / len(at_risk)


def route_pairs(pairs):
    """Project category pairs onto the routing decision the agent actually takes."""
    return [(route_for(gold), route_for(predicted)) for gold, predicted in pairs]


def escalation_rate(pairs):
    """How often the agent would hand off to a human, by predicted route."""
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(1 for _, predicted in pairs if route_for(predicted) is Route.ESCALATE) / len(pairs)


def _fmt_row(cells, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()


def format_report(pairs, title="intake classification"):
    pairs = list(pairs)
    metrics = per_class_metrics(pairs)
    lines = [f"== {title} ==", f"n = {len(pairs)}", ""]

    header = ["class", "prec", "recall", "f1", "support", "predicted"]
    widths = [18, 6, 6, 6, 8, 9]
    lines.append(_fmt_row(header, widths))
    lines.append(_fmt_row(["-" * w for w in widths], widths))
    for label in sorted(metrics, key=lambda c: -metrics[c]["support"]):
        m = metrics[label]
        value = label.value if isinstance(label, Category) else str(label)
        lines.append(_fmt_row(
            [value, f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}",
             m["support"], m["predicted"]],
            widths,
        ))

    lines += [
        "",
        f"accuracy            {accuracy(pairs):.3f}",
        f"macro F1            {macro_f1(pairs):.3f}",
        f"escalation rate     {escalation_rate(pairs):.3f}",
        f"UNSAFE ANSWER RATE  {unsafe_answer_rate(pairs):.3f}   "
        f"(must-not-answer issues routed to ANSWER - keep at 0)",
    ]

    routes = route_pairs(pairs)
    lines += ["", "-- routing view --", _fmt_row(header, widths),
              _fmt_row(["-" * w for w in widths], widths)]
    route_metrics = per_class_metrics(routes)
    for label in sorted(route_metrics, key=lambda r: -route_metrics[r]["support"]):
        m = route_metrics[label]
        value = label.value if isinstance(label, Route) else str(label)
        lines.append(_fmt_row(
            [value, f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}",
             m["support"], m["predicted"]],
            widths,
        ))
    return "\n".join(lines)


def worst_confusions(pairs, limit=8):
    """Most frequent (gold -> predicted) mistakes, for deriving the next fix."""
    wrong = [(g, p) for g, p in pairs if g != p]
    counts = defaultdict(int)
    for gold, predicted in wrong:
        counts[(gold, predicted)] += 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
