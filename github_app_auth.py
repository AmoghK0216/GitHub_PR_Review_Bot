from pathlib import Path
from os import getenv

from github import Github, GithubIntegration


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


def get_pr_changed_files(installation_id: int, repo_full_name: str, pr_number: int) -> list[dict[str, str | None]]:
    client = get_installation_client(installation_id)
    pull_request = client.get_repo(repo_full_name).get_pull(pr_number)
    files: list[dict[str, str | None]] = []

    for pr_file in pull_request.get_files():
        files.append(
            {
                "filename": pr_file.filename,
                "status": pr_file.status,
                "patch": pr_file.patch,
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
