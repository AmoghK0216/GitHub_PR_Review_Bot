import hashlib
import hmac
from os import getenv

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from github_app_auth import get_pr_changed_files, github_app_is_configured, post_pr_comment

load_dotenv()

app = FastAPI(title="PR Review Bot")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Backend is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request) -> dict:
    body = await request.body()

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not _is_valid_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event = request.headers.get("X-GitHub-Event", "unknown")
    delivery_id = request.headers.get("X-GitHub-Delivery", "unknown")

    if event != "pull_request":
        return {
            "status": "ignored",
            "reason": "unsupported_event",
            "event": event,
            "delivery_id": delivery_id,
        }

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    action = payload.get("action", "unknown")
    allowed_actions = {"opened", "synchronize", "reopened"}
    if action not in allowed_actions:
        return {
            "status": "ignored",
            "reason": "unsupported_action",
            "event": event,
            "action": action,
            "delivery_id": delivery_id,
        }

    installation = payload.get("installation", {}) if isinstance(payload.get("installation"), dict) else {}
    repository = payload.get("repository", {}) if isinstance(payload.get("repository"), dict) else {}
    pull_request = payload.get("pull_request", {}) if isinstance(payload.get("pull_request"), dict) else {}

    installation_id = installation.get("id")
    repo_name = repository.get("full_name")
    pr_number = pull_request.get("number") or payload.get("number")

    if not installation_id or not repo_name or not pr_number:
        return {
            "status": "ignored",
            "reason": "missing_required_fields",
            "event": event,
            "action": action,
            "delivery_id": delivery_id,
        }

    github_ready, github_error = github_app_is_configured()
    if not github_ready:
        return {
            "status": "blocked",
            "reason": "github_app_not_configured",
            "event": event,
            "action": action,
            "delivery_id": delivery_id,
            "repo": repo_name,
            "pr_number": pr_number,
            "details": github_error,
        }

    try:
        changed_files = get_pr_changed_files(
            installation_id=int(installation_id),
            repo_full_name=str(repo_name),
            pr_number=int(pr_number),
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": "github_api_error",
            "event": event,
            "action": action,
            "delivery_id": delivery_id,
            "repo": repo_name,
            "pr_number": pr_number,
            "details": str(exc),
        }

    try:
        comment_body = (
            "✅ Webhook verified.\n\n"
            f"Action: `{action}`\n"
            f"Changed files detected: **{len(changed_files)}**"
        )
        post_pr_comment(
            installation_id=int(installation_id),
            repo_full_name=str(repo_name),
            pr_number=int(pr_number),
            body=comment_body,
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": "github_comment_error",
            "event": event,
            "action": action,
            "delivery_id": delivery_id,
            "repo": repo_name,
            "pr_number": pr_number,
            "details": str(exc),
        }

    return {
        "status": "queued",
        "event": event,
        "action": action,
        "delivery_id": delivery_id,
        "installation_id": installation_id,
        "repo": repo_name,
        "pr_number": pr_number,
        "changed_files_count": len(changed_files),
        "changed_files": changed_files[:15],
    }


def _is_valid_signature(body: bytes, signature: str) -> bool:
    secret = getenv("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        return False
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
