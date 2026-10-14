"""Optional single-call client for a chat-completions-compatible endpoint."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def complete(prompt: str, *, base_url: str | None = None,
             model: str | None = None,
             api_key_env: str = "FORMALJUDGE_API_KEY") -> tuple[str, str]:
    base = (base_url if base_url is not None else os.environ.get("FORMALJUDGE_BASE_URL", "")).rstrip("/")
    model = model if model is not None else os.environ.get("FORMALJUDGE_MODEL", "")
    token = os.environ.get(api_key_env, "")
    url = urlsplit(base)
    if not model or not base:
        raise ValueError("Set FORMALJUDGE_BASE_URL and FORMALJUDGE_MODEL")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("Use a base URL without embedded credentials, query, or fragment")
    if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")):
        raise ValueError("Use HTTPS, or HTTP for a local model server")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}).encode()
    request = Request(base + "/chat/completions", data=payload, headers=headers, method="POST")
    try:
        with build_opener(NoRedirects()).open(request, timeout=120) as response:
            data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError("Model response exceeds the sample tool size limit")
        body = json.loads(data)
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model returned no text")
        return content, model
    except HTTPError as exc:
        code = exc.code
        exc.close()
        raise RuntimeError(f"Model endpoint returned HTTP {code}; no verdict produced") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Model endpoint unavailable; no verdict produced") from None
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        raise RuntimeError("Model endpoint returned an unsupported response") from None
