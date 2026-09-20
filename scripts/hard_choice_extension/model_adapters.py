from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hard_choice_common import MODELS, RUNTIME, SERVER


def sanitize(value: Any, key: str = "") -> str:
    text = str(value).replace(key, "[REDACTED]") if key else str(value)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?\S+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)\S+", r"\1[REDACTED]", text)
    return text[:2000]


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 180) -> tuple[dict[str, Any] | None, float, str | None, int | None]:
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json", **(headers or {})})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), time.perf_counter() - start, None, response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return None, time.perf_counter() - start, body, exc.code
    except Exception as exc:
        return None, time.perf_counter() - start, str(exc), None


class LocalAdapter:
    def __init__(self, model_key: str, log_path: Path):
        self.model_key = model_key
        self.cfg = MODELS[model_key]
        self.log_path = log_path
        self.log_handle = None
        self.process = None

    def __enter__(self):
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        cmd = [str(SERVER), "--model", str(self.cfg["path"]), "--ctx-size", "4096", "--n-gpu-layers", "99", "--flash-attn", "on", "--parallel", str(self.cfg["parallel"]), "--host", "127.0.0.1", "--port", str(self.cfg["port"]), "--no-webui", "--reasoning", "off", "--reasoning-budget", "0", "--chat-template-kwargs", '{"enable_thinking":false}', "--seed", "2026"]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(cmd, cwd=RUNTIME, stdout=self.log_handle, stderr=subprocess.STDOUT, creationflags=flags)
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.model_key} server exited {self.process.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.cfg['port']}/health", timeout=3) as response:
                    if json.loads(response.read().decode())["status"] == "ok":
                        return self
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"{self.model_key} load timeout")

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {"model": self.cfg["display"], "messages": messages, "temperature": 0, "top_p": 1.0, "top_k": 1, "seed": 2026, "max_tokens": 4, "stream": False, "cache_prompt": False}
        response, latency, error, status = post_json(f"http://127.0.0.1:{self.cfg['port']}/v1/chat/completions", payload)
        raw = None
        returned = None
        usage = None
        if response:
            returned = response.get("model")
            choices = response.get("choices") or []
            raw = ((choices[0].get("message") or {}).get("content") if choices else None)
            usage = response.get("usage")
        return {"raw": raw, "latency_seconds": latency, "error": sanitize(error) if error else None, "http_status": status, "returned_model_id": returned, "usage": usage, "config": payload | {"messages": "[omitted]"}}

    def __exit__(self, *_):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log_handle:
            self.log_handle.close()


class MotifAdapter:
    def __init__(self, max_completion_tokens: int = 4):
        self.cfg = MODELS["motif"]
        self.max_completion_tokens = max_completion_tokens
        self.key = os.environ.get("MORPH_API_KEY")
        if not self.key:
            raise RuntimeError("MORPH_API_KEY is not present")

    def __enter__(self):
        return self

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {"model": self.cfg["model_id"], "messages": messages, "temperature": 0, "top_p": 1.0, "seed": 2026, "max_completion_tokens": self.max_completion_tokens, "stream": False}
        headers = {"Authorization": "Bearer " + self.key}
        response, latency, error, status = post_json(self.cfg["base_url"] + "/chat/completions", payload, headers=headers, timeout=90)
        raw = None
        returned = None
        usage = None
        if response:
            returned = response.get("model")
            choices = response.get("choices") or []
            message = (choices[0].get("message") or {}) if choices else {}
            raw = message.get("content")
            usage = response.get("usage")
        return {"raw": raw, "reasoning": message.get("reasoning") if response else None, "latency_seconds": latency, "error": sanitize(error, self.key) if error else None, "http_status": status, "returned_model_id": returned, "usage": usage, "config": payload | {"messages": "[omitted]"}}

    def __exit__(self, *_):
        self.key = ""


def adapter_for(model_key: str, log_path: Path):
    return MotifAdapter() if model_key == "motif" else LocalAdapter(model_key, log_path)
