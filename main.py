import hashlib
import hmac
from os import getenv

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

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

    repository = payload.get("repository", {}) if isinstance(payload.get("repository"), dict) else {}
    pull_request = payload.get("pull_request", {}) if isinstance(payload.get("pull_request"), dict) else {}

    repo_name = repository.get("full_name", "unknown")
    pr_number = payload.get("number") or pull_request.get("number")

    return {
        "status": "queued",
        "event": event,
        "action": action,
        "delivery_id": delivery_id,
        "repo": repo_name,
        "pr_number": pr_number,
    }


def _is_valid_signature(body: bytes, signature: str) -> bool:
    secret = getenv("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        return False
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
