"""Caption-leakage checker served through a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import requests
from jinja2 import Template


try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


_BOOL_TRUE = {"1", "true", "yes", "y", "on", "t"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _BOOL_TRUE


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _healthcheck_servers(ip: str, ports: list[int], timeout_s: int = 120, api_key: str = "") -> None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for p in ports:
        url = f"http://{ip}:{p}/health"
        try:
            r = requests.get(url, headers=headers, timeout=timeout_s)
        except Exception as e:
            raise RuntimeError(f"[vllm_judge] healthcheck failed: url={url} err={e}") from e

        if r.status_code != 200:
            raise RuntimeError(
                f"[vllm_judge] healthcheck failed: url={url} status={r.status_code} body={r.text[:200]}"
            )


def _safe_to_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return repr(x)


def _encode_pil_to_data_url(img: "Image.Image", fmt: str = "PNG") -> str:
    """Encode a PIL image as a data URL for OpenAI-style image_url."""
    if Image is None:
        raise RuntimeError("PIL is not available, cannot encode images for vllm judge")

    if img.mode not in {"RGB", "RGBA"}:
        img = img.convert("RGB")

    buf = BytesIO()
    fmt = fmt.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _open_image_any(obj: Any) -> Optional["Image.Image"]:
    """Open an image from a path, bytes, mapping, or PIL object."""
    if Image is None:
        return None

    if obj is None:
        return None

    if hasattr(obj, "save") and hasattr(obj, "mode"):
        return obj

    try:
        if isinstance(obj, str):
            im = Image.open(obj)
            try:
                im.load()
                out = im.copy()
            finally:
                im.close()
            return out
        if isinstance(obj, bytes):
            im = Image.open(BytesIO(obj))
            try:
                im.load()
                out = im.copy()
            finally:
                im.close()
            return out
        if isinstance(obj, dict) and "bytes" in obj:
            im = Image.open(BytesIO(obj["bytes"]))
            try:
                im.load()
                out = im.copy()
            finally:
                im.close()
            return out
    except Exception:
        return None

    return None


def extract_pil_images(multi_modal_data: Any, limit: int = 4) -> list["Image.Image"]:
    """Extract up to `limit` PIL images from a multi_modal_data field."""
    if Image is None:
        return []

    imgs: list["Image.Image"] = []
    if not multi_modal_data:
        return imgs

    try:
        if isinstance(multi_modal_data, dict):
            processed = multi_modal_data.get("processed_images", None)
            if processed is not None:
                for it in processed:
                    im = _open_image_any(it)
                    if im is not None:
                        imgs.append(im)
                        if len(imgs) >= limit:
                            return imgs
                if imgs:
                    return imgs

            # Fall back to raw images.
            raw = multi_modal_data.get("images", None)
            if raw is not None:
                for it in raw:
                    im = _open_image_any(it)
                    if im is not None:
                        imgs.append(im)
                        if len(imgs) >= limit:
                            return imgs
    except Exception:
        return imgs

    return imgs


@dataclass
class VLLMJudgeConfig:
    ip: str
    ports: tuple[int, ...] = (9100, 9101, 9102, 9103)
    model: str = ""  # empty => auto-detect via /v1/models when possible
    template_path: str = "./prompts/judge.jinja"
    request_timeout_s: int = 180
    max_retries: int = 3
    max_concurrency_per_server: int = 128
    max_images: int = 4
    use_images: bool = True
    image_format: str = "PNG"
    api_key: str = ""  # optional


class VLLMCaptionJudge:
    """A small client for a pool of vLLM OpenAI-compatible servers."""

    def __init__(self, cfg: VLLMJudgeConfig):
        self.cfg = cfg
        self.endpoints = [f"http://{cfg.ip}:{p}/v1/chat/completions" for p in cfg.ports]
        # requests.Session is not guaranteed to be thread-safe; keep one session per thread
        self._session_local = threading.local()
        self._template = self._load_template(cfg.template_path)
        self._model = cfg.model.strip() if cfg.model else ""
        self._headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            self._headers["Authorization"] = f"Bearer {cfg.api_key}"

        if not self._model:
            self._model = self._auto_detect_model() or ""

        self._log_every = 100
        log_dir = os.getenv("PRCO_LOG_DIR", os.path.join("outputs", "prco_7b", "logs"))
        self._log_path = os.path.join(log_dir, "vllm_judge.log")
        exp_name = os.getenv("EXPERIMENT_NAME", "").strip()
        if exp_name:
            exp_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", exp_name)
        exp_suffix = f"_{exp_name}" if exp_name else ""
        self._score_zero_log_path = os.path.join(log_dir, "judge", f"score_zero{exp_suffix}.log")
        self._log_lock = threading.Lock()
        self._score_zero_log_lock = threading.Lock()
        self._judge_count = 0

    @staticmethod
    def _load_template(path: str) -> Template:
        with open(path, encoding="utf-8") as f:
            return Template(f.read())

    def _get_session(self) -> requests.Session:
        """Get a thread-local requests session."""
        sess = getattr(self._session_local, "session", None)
        if sess is None:
            sess = requests.Session()
            self._session_local.session = sess
        return sess


    def _auto_detect_model(self) -> Optional[str]:
        """Detect the served model ID through /v1/models."""
        url = self.endpoints[0].rsplit("/v1/chat/completions", 1)[0] + "/v1/models"
        try:
            r = self._get_session().get(url, headers=self._headers, timeout=10)
            if r.status_code != 200:
                print(f"[vllm_judge] WARN /v1/models status={r.status_code} url={url} body={r.text[:200]}")
                return None
            data = r.json()
            models = data.get("data", [])
            if not models:
                print(f"[vllm_judge] WARN /v1/models returned empty list url={url}")
                return None
            mid = models[0].get("id", None)
            if mid:
                print(f"[vllm_judge] Auto-detected model id='{mid}' from {url}")
            return mid
        except Exception as e:
            print(f"[vllm_judge] WARN failed to auto-detect model via {url}: {e}")
            return None

    def _build_payload(
        self, question: str, caption: str, images: list["Image.Image"]
    ) -> tuple[dict[str, Any], str]:
        prompt_text = self._template.render(question=question, caption=caption)

        content: list[dict[str, Any]] = []
        for img in images:
            try:
                url = _encode_pil_to_data_url(img, fmt=self.cfg.image_format)
                content.append({"type": "image_url", "image_url": {"url": url}})
            except Exception as e:
                # If image encoding fails, continue with text only.
                print(f"[vllm_judge] WARN failed to encode image: {e}")

        content.append({"type": "text", "text": prompt_text})

        payload: dict[str, Any] = {
            "model": self._model or "",
            "messages": [
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1500,
        }

        # Some servers may not require/accept an empty model name.
        if not payload["model"]:
            payload.pop("model", None)

        return payload, prompt_text

    @staticmethod
    def _extract_text_from_response(resp_json: dict[str, Any]) -> str:
        try:
            choices = resp_json.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                if isinstance(msg, dict) and "content" in msg:
                    return _safe_to_str(msg.get("content", ""))
                if "text" in choices[0]:
                    return _safe_to_str(choices[0].get("text", ""))
        except Exception:
            pass
        return _safe_to_str(resp_json)

    @staticmethod
    def _parse_final_label(text: str) -> Optional[int]:
        """Parse the final 0/1 label from the model output."""
        s = text.strip()
        # Prefer JSON output: {"label":0/1,"reason":"..."}
        try:
            if "{" in s and "}" in s:
                start = s.find("{")
                end = s.rfind("}")
                if start >= 0 and end > start:
                    obj = json.loads(s[start : end + 1])
                    if isinstance(obj, dict) and "label" in obj:
                        lb = obj.get("label")
                        if isinstance(lb, str):
                            lb = lb.strip()
                        if lb in (0, 1, "0", "1"):
                            return int(lb)
        except Exception:
            pass

        m = re.search(r"([01])\s*$", s)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _extract_reason_and_score(
        text: Optional[str], label: Optional[int]
    ) -> tuple[str, Optional[int]]:
        s = _safe_to_str(text).strip()
        if not s:
            return "", label
        try:
            if "{" in s and "}" in s:
                start = s.find("{")
                end = s.rfind("}")
                if start >= 0 and end > start:
                    obj = json.loads(s[start : end + 1])
                    if isinstance(obj, dict):
                        reason = obj.get("reason")
                        if reason is None:
                            reason = obj.get("explanation") or obj.get("rationale") or ""
                        score = obj.get("label", obj.get("score", label))
                        if isinstance(score, str):
                            score = score.strip()
                        if score in ("0", "1"):
                            score = int(score)
                        elif isinstance(score, (int, float)):
                            score = int(score)
                        reason_s = _safe_to_str(reason).strip()
                        if not reason_s:
                            reason_s = s
                        return reason_s, score
        except Exception:
            pass
        return s, label

    def _maybe_log_score_zero(
        self,
        *,
        prompt_text: str,
        response_text: Optional[str],
        label: Optional[int],
    ) -> None:
        if label != 0:
            return
        reason, score = self._extract_reason_and_score(response_text, label)
        entry = {
            "ts": time.time(),
            "prompt_text": prompt_text,
            "reason": reason,
            "score": score,
        }
        os.makedirs(os.path.dirname(self._score_zero_log_path), exist_ok=True)
        with self._score_zero_log_lock:
            with open(self._score_zero_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _maybe_log_judge(
        self,
        *,
        index: int,
        question: str,
        caption: str,
        label: Optional[int],
        error: Optional[str],
        response_text: Optional[str],
    ) -> None:
        if self._log_every <= 0:
            return
        with self._log_lock:
            self._judge_count += 1
            if self._judge_count % self._log_every != 0:
                return
            entry = {
                "ts": time.time(),
                "count": self._judge_count,
                "question": question,
                "caption": caption,
                "label": label,
                "error": error,
                "response_text": response_text,
            }
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _post_with_retries(
        self, url: str, payload: dict[str, Any]
    ) -> tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        """Return (label, error_message, response_text, raw_response)."""
        last_err: Optional[str] = None
        last_resp_text: Optional[str] = None
        last_raw_response: Optional[str] = None
        for attempt in range(1, max(self.cfg.max_retries, 1) + 1):
            try:
                r = self._get_session().post(
                    url,
                    headers=self._headers,
                    data=json.dumps(payload),
                    timeout=self.cfg.request_timeout_s,
                )
                last_raw_response = r.text
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code} body={r.text[:300]}"
                    raise RuntimeError(last_err)

                try:
                    resp_json = r.json()
                except Exception:
                    resp_json = {}
                text = self._extract_text_from_response(resp_json)
                last_resp_text = text
                label = self._parse_final_label(text)
                if label is None:
                    last_err = f"Could not parse 0/1 from output: {text[-200:]}"
                    raise RuntimeError(last_err)
                return label, None, last_resp_text, last_raw_response
            except Exception as e:
                last_err = f"attempt={attempt}/{self.cfg.max_retries} url={url} err={e}"
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

        return None, last_err, last_resp_text, last_raw_response

    def judge_batch(
        self,
        questions: list[str],
        captions: list[str],
        multi_modal_data_list: list[Any],
    ) -> tuple[list[Optional[int]], list[str]]:
        """Judge a batch. Returns (labels, error_messages).

        - labels[i] is 0 or 1 on success, or None on failure.
        - error_messages contains a few representative errors for logging.
        """
        n = len(captions)
        if len(questions) != n or len(multi_modal_data_list) != n:
            raise ValueError(
                f"judge_batch expects aligned lengths, got questions={len(questions)} captions={len(captions)} "
                f"multi_modal_data={len(multi_modal_data_list)}"
            )

        endpoints = self.endpoints
        k = max(len(endpoints), 1)
        per_server = max(self.cfg.max_concurrency_per_server, 1)
        max_workers = min(n, per_server * k)
        if max_workers <= 0:
            max_workers = 1

        labels: list[Optional[int]] = [None] * n
        errors: list[str] = []

        def _task(i: int) -> tuple[int, Optional[int], Optional[str]]:
            url = endpoints[i % k]
            if self.cfg.use_images:
                images = extract_pil_images(multi_modal_data_list[i], limit=self.cfg.max_images)
            else:
                images = []
            payload, prompt_text = self._build_payload(questions[i], captions[i], images)
            label, err, resp_text, raw_response = self._post_with_retries(url, payload)
            self._maybe_log_judge(
                index=i,
                question=questions[i],
                caption=captions[i],
                label=label,
                error=err,
                response_text=resp_text,
            )
            self._maybe_log_score_zero(prompt_text=prompt_text, response_text=resp_text, label=label)
            return i, label, err

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_task, i) for i in range(n)]
            for fut in as_completed(futs):
                i, label, err = fut.result()
                labels[i] = label
                if err is not None and len(errors) < 8:
                    errors.append(err)

        return labels, errors


def build_judge_from_env() -> Optional[VLLMCaptionJudge]:
    """Construct a judge client from environment variables."""

    enabled = env_flag("vllm_judge", default=False) or env_flag("VLLM_JUDGE", default=False)
    if not enabled:
        return None

    ip = env_str("vllm_judge_ip", default="") or env_str("VLLM_JUDGE_IP", default="")
    if not ip:
        print("[vllm_judge] WARN vllm_judge is enabled but no IP provided (vllm_judge_ip / VLLM_JUDGE_IP). Skip.")
        return None

    ports_s = env_str("vllm_judge_ports", default="9100,9101,9102,9103")
    ports: list[int] = []
    for p in ports_s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            ports.append(int(p))
        except Exception:
            continue
    if not ports:
        ports = [9100, 9101, 9102, 9103]

    api_key = env_str("vllm_judge_api_key", default="") or env_str("VLLM_JUDGE_API_KEY", default="")
    use_images_str = env_str("vllm_judge_use_images", default="")
    if not use_images_str:
        use_images_str = env_str("VLLM_JUDGE_USE_IMAGES", default="")
    if use_images_str == "":
        use_images = True
    else:
        use_images = use_images_str.strip().lower() in _BOOL_TRUE


    _healthcheck_servers(
        ip, ports, timeout_s=env_int("vllm_judge_healthcheck_timeout_s", default=120), api_key=api_key
    )

    cfg = VLLMJudgeConfig(
        ip=ip,
        ports=tuple(ports),
        model=env_str("vllm_judge_model", default="") or env_str("VLLM_JUDGE_MODEL", default=""),
        template_path=env_str("vllm_judge_template", default="")
        or env_str("VLLM_JUDGE_TEMPLATE", default="./prompts/judge.jinja"),
        request_timeout_s=env_int("vllm_judge_timeout_s", default=180),
        max_retries=env_int("vllm_judge_max_retries", default=3),
        max_concurrency_per_server=env_int("vllm_judge_max_concurrency_per_server", default=128),
        use_images=use_images,
        max_images=env_int("vllm_judge_max_images", default=env_int("VLLM_JUDGE_MAX_IMAGES", default=4)),
        image_format=env_str("vllm_judge_image_format", default="PNG"),
        api_key=api_key,
    )

    try:
        return VLLMCaptionJudge(cfg)
    except Exception as e:
        print(f"[vllm_judge] ERROR failed to init VLLMCaptionJudge: {e}")
        return None
