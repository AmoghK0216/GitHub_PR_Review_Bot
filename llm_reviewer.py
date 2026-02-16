from os import getenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_review_comment(changed_files: list[dict[str, str | None]]) -> str:
    review_input = _build_review_input(changed_files)
    if not review_input.strip():
        return "## 🤖 AI Code Review\n\nNo reviewable text patches were found in this PR."

    prompt = ChatPromptTemplate.from_template(
                """
        You are a senior code reviewer.
        Review only the provided PR diffs and return concise markdown.

        Use these sections exactly:
        1. Critical Issues
        2. Risks
        3. Suggestions
        4. Summary

        Rules:
        - Focus on changed code only.
        - If no critical issues are found, explicitly say: "No critical issues found."
        - Keep feedback specific and actionable.

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

    for item in changed_files[:max_files]:
        filename = item.get("filename") or "unknown"
        status = item.get("status") or "unknown"
        patch = item.get("patch") or ""

        if not patch.strip():
            continue

        parts.append(
            "\n".join(
                [
                    f"File: {filename}",
                    f"Status: {status}",
                    "Diff:",
                    patch[:3000],
                ]
            )
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
