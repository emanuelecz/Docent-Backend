import json

from core.config import get_settings
from ai.summary_llm import get_anthropic_client
from evals.taxonomy import Category

settings = get_settings()

# Body text is truncated before classification: category is almost always decidable
# from the opening of an issue, and long tracebacks or log dumps cost tokens without
# adding signal.
MAX_BODY_CHARS = 4000


class ClassificationError(RuntimeError):
    pass


def _parse(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClassificationError(f"classifier did not return JSON: {raw[:200]!r}") from exc

    try:
        category = Category(payload["category"])
    except (KeyError, ValueError):
        # An out-of-set or missing category is treated as an abstention rather than an
        # error: UNKNOWN escalates, which is the behavior we want when the model has
        # gone off-script anyway.
        category = Category.UNKNOWN

    confidence = payload.get("confidence")
    return {
        "category": category,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
        "reason": str(payload.get("reason", ""))[:300],
    }


def classify_issue(title: str, original_question: str) -> dict:
    """Classify a raw issue into a triage Category.

    Deliberately takes only title and body. GitHub labels are the eval's ground truth
    and are usually absent when an issue is first opened, so the runtime path must
    never read them.
    """
    client = get_anthropic_client()
    system_prompt = settings.prompts.classify_system_prompt.format(
        repo=f"{settings.repo_owner}/{settings.repo_name}"
    )
    user_prompt = settings.prompts.classify_user_template.format(
        title=title or "",
        original_question=(original_question or "")[:MAX_BODY_CHARS],
    )

    resp = client.messages.create(
        model=settings.classify_model,
        temperature=0,
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return _parse(resp.content[0].text)
