from os import getenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_review_comment(changed_files: list[dict[str, str | None]]) -> str:
    api_key = getenv("GROQ_API_KEY", "").strip()
    model = getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    base_url = getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip().rstrip("/")

    if not api_key:
        return _fallback_comment(changed_files, "GROQ_API_KEY not configured")

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
        return f"## 🤖 AI Code Review\n\n{content.strip()}"
    except Exception as exc:
        return _fallback_comment(changed_files, f"Groq request failed: {exc}")


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
