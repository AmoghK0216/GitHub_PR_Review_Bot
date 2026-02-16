from pathlib import Path
import re
from os import getenv

from github import Github, GithubIntegration

HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
HEADER_CONTEXT_LINES = 120
WINDOW_CONTEXT_RADIUS = 40
MAX_HUNK_WINDOWS = 3
MAX_SUPPORT_CONTEXT_CHARS = 6000


def get_github_integration() -> GithubIntegration:
    app_id = getenv("GITHUB_APP_ID", "").strip()
    private_key_path = getenv("GITHUB_PRIVATE_KEY_PATH", "").strip()

    if not app_id:
        raise ValueError("GITHUB_APP_ID is not set")
    if not private_key_path:
        raise ValueError("GITHUB_PRIVATE_KEY_PATH is not set")

    key_path = Path(private_key_path)
    if not key_path.exists():
        raise ValueError(f"Private key not found: {private_key_path}")

    private_key = key_path.read_text(encoding="utf-8")
    return GithubIntegration(app_id, private_key)


def github_app_is_configured() -> tuple[bool, str]:
    try:
        get_github_integration()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def get_installation_client(installation_id: int) -> Github:
    integration = get_github_integration()
    token = integration.get_access_token(installation_id).token
    return Github(token)


def _extract_new_file_hunks(patch: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for line in patch.splitlines():
        match = HUNK_HEADER_RE.match(line)
        if not match:
            continue

        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) else 1
        if count <= 0:
            count = 1

        ranges.append((start, start + count - 1))

    return ranges


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda item: item[0])
    merged: list[tuple[int, int]] = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _render_line_block(lines: list[str], start_line: int, end_line: int, title: str) -> str:
    numbered_lines = [f"{line_no:>5}: {lines[line_no - 1]}" for line_no in range(start_line, end_line + 1)]
    return "\n".join([title, *numbered_lines])


def _get_file_text(repo, filename: str, ref: str) -> str | None:
    try:
        content_file = repo.get_contents(filename, ref=ref)
    except Exception:
        return None

    if isinstance(content_file, list):
        return None

    try:
        data = content_file.decoded_content
    except Exception:
        return None

    return data.decode("utf-8", errors="replace")


def _build_support_context(file_text: str, patch: str | None) -> str | None:
    lines = file_text.splitlines()
    if not lines:
        return None

    sections: list[str] = []

    header_end = min(len(lines), HEADER_CONTEXT_LINES)
    sections.append(_render_line_block(lines, 1, header_end, "Header Context (line-numbered):"))

    if patch and patch.strip():
        hunk_ranges = _extract_new_file_hunks(patch)
        expanded_ranges: list[tuple[int, int]] = []

        for start, end in hunk_ranges[:MAX_HUNK_WINDOWS]:
            expanded_start = max(1, start - WINDOW_CONTEXT_RADIUS)
            expanded_end = min(len(lines), end + WINDOW_CONTEXT_RADIUS)
            expanded_ranges.append((expanded_start, expanded_end))

        for index, (start, end) in enumerate(_merge_ranges(expanded_ranges)[:MAX_HUNK_WINDOWS], start=1):
            sections.append(_render_line_block(lines, start, end, f"Nearby Context Window {index} (line-numbered):"))

    support_context = "\n\n".join(sections).strip()
    if not support_context:
        return None

    return support_context[:MAX_SUPPORT_CONTEXT_CHARS]


def get_pr_changed_files(installation_id: int, repo_full_name: str, pr_number: int) -> list[dict[str, str | None]]:
    client = get_installation_client(installation_id)
    repo = client.get_repo(repo_full_name)
    pull_request = repo.get_pull(pr_number)
    head_sha = pull_request.head.sha
    files: list[dict[str, str | None]] = []

    for pr_file in pull_request.get_files():
        support_context: str | None = None
        if pr_file.status != "removed":
            file_text = _get_file_text(repo, pr_file.filename, head_sha)
            if file_text is not None:
                support_context = _build_support_context(file_text, pr_file.patch)

        files.append(
            {
                "filename": pr_file.filename,
                "status": pr_file.status,
                "patch": pr_file.patch,
                "support_context": support_context,
            }
        )

    return files


def post_pr_comment(installation_id: int, repo_full_name: str, pr_number: int, body: str) -> None:
    client = get_installation_client(installation_id)
    pull_request = client.get_repo(repo_full_name).get_pull(pr_number)
    pull_request.create_issue_comment(body)


def upsert_pr_comment(
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
    body: str,
    marker: str = "<!-- ai-pr-review-bot -->",
) -> None:
    client = get_installation_client(installation_id)
    pull_request = client.get_repo(repo_full_name).get_pull(pr_number)
    content = f"{marker}\n{body}"

    for comment in pull_request.get_issue_comments():
        if marker in comment.body:
            comment.edit(content)
            return

    pull_request.create_issue_comment(content)
