from os import getenv

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
        Review ONLY the changed diff lines and return concise markdown.

        Use these sections exactly:
        1. Critical Issues
        2. Risks
        3. Suggestions
        4. Summary

        Scope Rules (strict):
        - Primary review target is ONLY lines in the "Changed Diff" sections.
        - "Support Context" is verification-only context. Do NOT review or critique unchanged code in support context.
        - Mention imports/config/defaults/function arguments only when directly tied to a changed diff line.
        - Do not invent regressions outside changed diff lines.

        Evidence Rules (strict):
        - Every issue/risk must include one short evidence quote from the provided input.
        - If evidence is insufficient, write exactly: "Unverified with provided context." and avoid prescriptive fixes.
        - If no critical issues are evidenced in changed diff lines, write exactly: "No critical issues found."
        - If any section is marked with "[TRUNCATED]", treat it as incomplete context and do NOT report syntax/compile errors from cut-off text.

        Output Rules:
        - Keep feedback specific and actionable, but only within changed diff scope.
        - Do not comment on deployment setup, env strategy, or architecture unless the changed diff lines directly modify them.
        - If no critical issues are found, explicitly say: "No critical issues found."

        PR Diff Content:
        {review_input}
        """.strip()
    )

    provider_errors: list[str] = []
    for provider in ["NAVIGATOR", "GROQ"]:
        content, err = _invoke_provider(provider, prompt, review_input)
        if content:
            return f"## 🤖 AI Code Review\n\n_Model: {provider.lower()}_\n\n{content.strip()}"
        if err:
            provider_errors.append(err)

    return _fallback_comment(changed_files, " | ".join(provider_errors) if provider_errors else "No provider configured")


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

        parts.append(
            "\n".join(file_parts)
        )

    return "\n\n---\n\n".join(parts)


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
