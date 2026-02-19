import json
import re
from os import getenv
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_review_comment(changed_files: list[dict[str, str | None]]) -> str:
    review_input = _build_review_input(changed_files)
    if not review_input.strip():
        return "## 🤖 AI Code Review\n\nNo reviewable code context was found in this PR."

    prompt = ChatPromptTemplate.from_template(
        """
You are a senior code reviewer.
Analyze only changed diff lines. Support Context is verification-only.

Return ONLY valid JSON. No markdown, no prose outside JSON.

Required JSON schema:
{{
    "critical_issues": [
        {{"file": "...", "claim": "...", "evidence": "...", "confidence": "high|medium|low"}}
    ],
    "risks": [
        {{"file": "...", "claim": "...", "evidence": "...", "confidence": "high|medium|low"}}
    ],
    "suggestions": [
        {{"file": "...", "claim": "...", "evidence": "...", "confidence": "high|medium|low"}}
    ],
    "summary": "one short sentence"
}}

Rules:
- Scope is only changed diff lines.
- Use support context only to verify claims about changed diff lines.
- Every item must include file, claim, evidence, confidence.
- If evidence is weak, set confidence to low.
- Never invent files outside changed files.

Input:
{review_input}
""".strip()
    )

    provider_errors: list[str] = []
    for provider in ["NAVIGATOR", "GROQ"]:
        content, err = _invoke_provider(provider, prompt, review_input)
        if content:
            parsed = _parse_json_payload(content)
            if parsed is None:
                provider_errors.append(f"{provider}: invalid JSON response")
                continue

            validated = _validate_structured_review(parsed, review_input, changed_files)
            rendered = _render_review(validated)
            return f"## 🤖 AI Code Review\n\n_Model: {provider.lower()}_\n\n{rendered}"

        if err:
            provider_errors.append(err)

    reason = " | ".join(provider_errors) if provider_errors else "No provider configured"
    return _fallback_comment(changed_files, reason)


def _invoke_provider(
    provider: str,
    prompt: ChatPromptTemplate,
    review_input: str,
) -> tuple[str | None, str | None]:
    api_key = getenv(f"{provider}_API_KEY", "").strip()
    base_url = getenv(f"{provider}_BASE_URL", "").strip().rstrip("/")
    model = getenv(f"{provider}_MODEL", "").strip()

    if not api_key:
        return None, f"{provider}: API key not configured"
    if not base_url:
        return None, f"{provider}: base URL not configured"
    if not model:
        return None, f"{provider}: model not configured"

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.2,
        timeout=45,
    )

    chain = prompt | llm | StrOutputParser()

    try:
        content = chain.invoke({"review_input": review_input})
        return content, None
    except Exception as exc:
        return None, f"{provider}: request failed: {exc}"


def _build_review_input(changed_files: list[dict[str, str | None]]) -> str:
    parts: list[str] = []
    max_files = 4
    max_patch_chars = 3000
    max_support_chars = 3500

    for item in changed_files[:max_files]:
        filename = item.get("filename") or "unknown"
        status = item.get("status") or "unknown"
        patch = item.get("patch") or ""
        support_context = item.get("support_context") or ""

        if not patch.strip() and not support_context.strip():
            continue

        file_parts = [
            f"File: {filename}",
            f"Status: {status}",
        ]

        if patch.strip():
            file_parts.extend(
                [
                    "Changed Diff (review scope):",
                    _truncate_for_prompt(patch, max_patch_chars),
                ]
            )
        else:
            file_parts.append("Changed Diff (review scope): Not available for this file (possibly binary/large).")

        if support_context.strip():
            file_parts.extend(
                [
                    "Support Context (verification-only, non-diff, line-numbered):",
                    _truncate_for_prompt(support_context, max_support_chars),
                ]
            )

        parts.append("\n".join(file_parts))

    return "\n\n---\n\n".join(parts)


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    marker = "\n[TRUNCATED]"
    allowed = max_chars - len(marker)
    if allowed <= 0:
        return marker.strip()

    chunk = text[:allowed]
    last_newline = chunk.rfind("\n")
    if last_newline > 0:
        chunk = chunk[:last_newline]

    return chunk.rstrip() + marker


def _parse_json_payload(model_output: str) -> dict[str, Any] | None:
    cleaned = model_output.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, dict) else None


def _validate_structured_review(
    parsed: dict[str, Any],
    review_input: str,
    changed_files: list[dict[str, str | None]],
) -> dict[str, list[str] | str]:
    diff_corpus = _extract_changed_diff_corpus(review_input)
    allowed_files = _build_allowed_file_names(changed_files)

    critical = _validate_section(parsed.get("critical_issues"), "critical", diff_corpus, allowed_files)
    risks = _validate_section(parsed.get("risks"), "risk", diff_corpus, allowed_files)
    suggestions = _validate_section(parsed.get("suggestions"), "suggestion", diff_corpus, allowed_files)

    if not critical:
        critical = ["No critical issues found."]

    summary_text = _build_summary_text(critical, risks, suggestions)

    return {
        "critical_issues": critical,
        "risks": risks,
        "suggestions": suggestions,
        "summary": summary_text,
    }


def _validate_section(
    section_data: Any,
    section_kind: str,
    diff_corpus: str,
    allowed_files: set[str],
) -> list[str]:
    if not isinstance(section_data, list):
        return []

    validated: list[str] = []
    for raw_item in section_data:
        item = _coerce_item(raw_item)
        if item is None:
            continue

        file_name = item["file"]
        claim = item["claim"]
        evidence = item["evidence"]
        confidence = item["confidence"]

        if not _is_file_allowed(file_name, allowed_files):
            continue

        evidence_level = _evidence_level(evidence, diff_corpus)
        if section_kind in {"critical", "risk"} and evidence_level == "low":
            continue

        effective_confidence = confidence if confidence in {"high", "medium", "low"} else evidence_level
        if section_kind == "suggestion" and evidence_level == "low":
            effective_confidence = "low"

        if section_kind in {"critical", "risk"} and effective_confidence == "low":
            continue

        validated.append(
            f"- [{file_name}] {claim} (Evidence: \"{evidence}\", Confidence: {effective_confidence.capitalize()})"
        )

    deduped: list[str] = []
    for line in validated:
        if line not in deduped:
            deduped.append(line)

    return deduped


def _coerce_item(raw_item: Any) -> dict[str, str] | None:
    if not isinstance(raw_item, dict):
        return None

    file_name = str(raw_item.get("file", "")).strip()
    claim = str(raw_item.get("claim", "")).strip()
    evidence = str(raw_item.get("evidence", "")).strip()
    confidence = str(raw_item.get("confidence", "")).strip().lower()

    if not file_name or not claim or not evidence:
        return None

    return {
        "file": file_name,
        "claim": claim,
        "evidence": evidence,
        "confidence": confidence,
    }


def _extract_changed_diff_corpus(review_input: str) -> str:
    marker = "Changed Diff (review scope):"
    stop_markers = [
        "Support Context (verification-only, non-diff, line-numbered):",
        "\n\n---\n\n",
    ]

    chunks: list[str] = []
    cursor = 0
    while True:
        start = review_input.find(marker, cursor)
        if start == -1:
            break
        start += len(marker)

        next_positions = [pos for pos in (review_input.find(stop, start) for stop in stop_markers) if pos != -1]
        end = min(next_positions) if next_positions else len(review_input)

        chunks.append(review_input[start:end])
        cursor = end

    return "\n".join(chunks)


def _build_allowed_file_names(changed_files: list[dict[str, str | None]]) -> set[str]:
    allowed: set[str] = set()
    for item in changed_files:
        filename = (item.get("filename") or "").strip().lower()
        if not filename:
            continue
        allowed.add(filename)
        allowed.add(filename.split("/")[-1])
    return allowed


def _is_file_allowed(file_name: str, allowed_files: set[str]) -> bool:
    normalized = file_name.strip().lower()
    basename = normalized.split("/")[-1]
    return normalized in allowed_files or basename in allowed_files


def _evidence_level(evidence: str, diff_corpus: str) -> str:
    evidence_lower = evidence.lower().strip()
    diff_lower = diff_corpus.lower()

    if evidence_lower and evidence_lower in diff_lower:
        return "high"

    normalized_diff = _normalize_text(diff_lower)
    normalized_evidence = _normalize_text(evidence_lower)

    if normalized_evidence and normalized_evidence in normalized_diff:
        return "medium"

    evidence_tokens = _tokenize(normalized_evidence)
    diff_tokens = _tokenize(normalized_diff)
    if len(evidence_tokens) >= 4 and diff_tokens:
        overlap = len(evidence_tokens.intersection(diff_tokens)) / max(1, len(evidence_tokens))
        if overlap >= 0.65:
            return "medium"

    return "low"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_\s]", " ", text.lower())).strip()


def _tokenize(text: str) -> set[str]:
    return {token for token in text.split() if len(token) >= 3}


def _build_summary_text(critical: list[str], risks: list[str], suggestions: list[str]) -> str:
    critical_count = sum(1 for line in critical if "No critical issues found." not in line)
    risk_count = len(risks)
    suggestion_count = len(suggestions)

    if critical_count == 0 and risk_count == 0 and suggestion_count == 0:
        return "No high-confidence findings could be verified from the provided changed diff lines."

    return (
        "Validated diff-scoped findings: "
        f"{critical_count} critical, {risk_count} risks, {suggestion_count} suggestions."
    )


def _render_review(validated: dict[str, list[str] | str]) -> str:
    critical = validated.get("critical_issues", [])
    risks = validated.get("risks", [])
    suggestions = validated.get("suggestions", [])
    summary = validated.get("summary", "No additional summary provided.")

    critical_lines = critical if isinstance(critical, list) else ["No critical issues found."]
    risk_lines = risks if isinstance(risks, list) else []
    suggestion_lines = suggestions if isinstance(suggestions, list) else []
    summary_line = summary if isinstance(summary, str) and summary.strip() else "No additional summary provided."

    sections: list[str] = ["Critical Issues", *critical_lines]

    if risk_lines:
        sections.extend(["", "Risks", *risk_lines])

    if suggestion_lines:
        sections.extend(["", "Suggestions", *suggestion_lines])

    sections.extend(["", "Summary", summary_line])
    return "\n".join(sections).strip()


def _fallback_comment(changed_files: list[dict[str, str | None]], reason: str) -> str:
    filenames = [str(item.get("filename", "unknown")) for item in changed_files[:10]]
    file_list = "\n".join(f"- {name}" for name in filenames) if filenames else "- No files detected"

    return (
        "## 🤖 AI Code Review\n\n"
        "AI review is currently unavailable, posting file summary instead.\n\n"
        f"Reason: {reason}\n\n"
        "### Changed Files\n"
        f"{file_list}"
    )
