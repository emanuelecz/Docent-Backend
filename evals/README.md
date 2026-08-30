# evals

Measurement for the intake phase, scored against ground truth we did not invent.

## Why this exists

The hard question — "is this draft any good?" — needs deep Pydantic knowledge to
answer. The intake question — "what kind of issue is this?" — does not, because
Pydantic's maintainers already answered it: every ingested issue carries the labels
they applied, and `ingestion/issues.py` stores them in `tags`. That makes intake a
closed-set classification problem with free, expert-supplied gold labels.

It is also the phase most worth measuring first. If intake misroutes a bug report
into the answer path, nothing downstream can recover — the agent posts a confident
public reply on an issue it should have handed to a human.

## Running it

Look at the data before trusting any decision made about it:

```bash
python -m evals.label_distribution
```

This reports the category and route distribution over both tables, what share of
issues carry no category label, and any labels the taxonomy does not yet cover.

Then score the classifier:

```bash
python -m evals.run_classification --per-category 15
python -m evals.run_classification --source open --include-unknown   # routing safety
```

Every prediction is written to `evals/results/` (gitignored). The disagreements are
the point: read them.

## The numbers, and which one matters

- **accuracy** — headline only. The corpus is dominated by bugs, so a classifier that
  always guessed `bug` would score deceptively well.
- **macro F1** — treats every category equally. The small categories are the ones
  whose routing differs, so this is the honest quality number.
- **escalation rate** — how often a human gets involved. High is acceptable; the cost
  of escalating is a maintainer's glance.
- **unsafe answer rate** — the share of must-not-answer issues (`bug`, `security`,
  `unknown`) that would be routed to `answer`. **This is the number that matters,
  and its target is zero.** Every other metric on the page can look healthy while
  this one is quietly non-zero.

The asymmetry is deliberate: escalating a question that could have been answered
costs nothing, and posting a hallucinated reply on `pydantic/pydantic` costs a lot.
Tune for precision on `answer` and accept a high escalation rate.

## Categories and routes

Derived from the labels that actually exist in `pydantic/pydantic`. `change` is the
repo's own label for "suggested alteration, not a new feature nor a bug".

| Category | Route | Why |
|---|---|---|
| `question` | `answer` | A correct answer demonstrably exists in the resolved corpus |
| `duplicate` | `prior_art` | Retrieval, not judgement |
| `feature_request` | `prior_art` | No ground truth for "is this a good idea" — surface prior discussion, do not opine |
| `change` | `prior_art` | Same |
| `bug` | `escalate` | Needs reproduction against repo state |
| `documentation` | `escalate` | Low value, and cheap for a human |
| `security` | `escalate` | Never automated |
| `unknown` | `escalate` | No category label, or the classifier abstained |

Not answering feature requests is a scoping decision, not a gap: the answer to one is
a maintainer's product judgement, which no amount of retrieval produces.

## Notes

- The runtime classifier reads only title and body. Labels are gold, and are usually
  absent when an issue is first opened — check the coverage section of
  `label_distribution` on the `open` table to see how absent.
- `taxonomy.CATEGORY_PRIORITY` resolves issues carrying several category labels.
  `duplicate` outranking `bug` is a judgement call, documented in place; change it and
  re-run to see the cost.
- `taxonomy.py` and `metrics.py` are pure and dependency-free, so `tests/` runs with
  no database and no API key.

## Next

The same shape extends to the phases after intake:

- **usage questions** → held-out eval: drop a closed issue from the index, feed its
  `original_question` in, compare the draft against the maintainer's `fix_summary`.
- **duplicate / prior art** → retrieval metrics (Hit@k, MRR) against known duplicate
  links.
- **bug reports** → did it escalate? Binary.

Stratify by category when sampling. A `fix_summary` for a declined feature request is
a different kind of text than one for a usage question, and averaging across them
produces a number that means nothing.
