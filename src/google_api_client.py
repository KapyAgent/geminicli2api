"""
Google API Client - Handles all communication with Google's Gemini API.
This module is used by both OpenAI compatibility layer and native Gemini endpoints.
"""
import json
import logging
from datetime import datetime, timezone
from fastapi import Response
from fastapi.responses import StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx
from typing import Optional

from .auth import get_credentials, save_credentials, get_user_project_id, onboard_user
from .http_client import get_http_client
from .http_retry import RetryConfig, post_with_retry, retry_after_seconds, sleep_before_retry
from .upstream_quota_state import get_quota_block, record_upstream_429, format_duration
from .utils import get_user_agent
from .config import (
    CODE_ASSIST_ENDPOINT,
    DEFAULT_SAFETY_SETTINGS,
    GOOGLE_FALLBACK_TOKEN,
    get_base_model_name,
    is_search_model,
    get_thinking_budget,
    should_include_thoughts,
    UPSTREAM_CONNECT_TIMEOUT_S,
    UPSTREAM_READ_TIMEOUT_S,
    UPSTREAM_STREAM_READ_TIMEOUT_S,
    UPSTREAM_MAX_ATTEMPTS,
    UPSTREAM_BACKOFF_BASE_S,
    UPSTREAM_BACKOFF_MAX_S,
)
import asyncio


GOOGLE_AI_STUDIO_ENDPOINT = "https://generativelanguage.googleapis.com"
_AI_STUDIO_ALLOWED_HARM_CATEGORIES = frozenset(
    {
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    }
)


def _ai_studio_model_path(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    model_name = model_name.strip()
    if not model_name:
        return None
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    return model_name


def _ai_studio_url(*, model_path: str, is_streaming: bool) -> str:
    action = "streamGenerateContent" if is_streaming else "generateContent"
    url = f"{GOOGLE_AI_STUDIO_ENDPOINT}/v1beta/{model_path}:{action}"
    if is_streaming:
        url += "?alt=sse"
    return url


def _ai_studio_headers(api_key: str) -> dict:
    return {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }


def _sanitize_ai_studio_request(request: dict) -> dict:
    cleaned = dict(request or {})
    safety_settings = cleaned.get("safetySettings")
    if isinstance(safety_settings, list):
        filtered: list[dict] = []
        for item in safety_settings:
            if not isinstance(item, dict):
                continue
            category = item.get("category")
            threshold = item.get("threshold")
            if category in _AI_STUDIO_ALLOWED_HARM_CATEGORIES and threshold:
                filtered.append({"category": category, "threshold": threshold})
        if filtered:
            cleaned["safetySettings"] = filtered
        else:
            cleaned.pop("safetySettings", None)
    return cleaned


async def _send_ai_studio_request(payload: dict, is_streaming: bool) -> Response:
    api_key = (GOOGLE_FALLBACK_TOKEN or "").strip()
    model_path = _ai_studio_model_path(payload.get("model"))
    if not api_key or not model_path:
        return Response(
            content=json.dumps({"error": {"message": "AI Studio fallback not configured.", "type": "api_error", "code": 500}}),
            status_code=500,
            media_type="application/json",
        )

    url = _ai_studio_url(model_path=model_path, is_streaming=is_streaming)
    headers = _ai_studio_headers(api_key)
    body = json.dumps(_sanitize_ai_studio_request(payload.get("request") or {}))
    client = get_http_client()

    if is_streaming:
        async def stream_generator():
            timeout = httpx.Timeout(timeout=None, connect=UPSTREAM_CONNECT_TIMEOUT_S, read=UPSTREAM_STREAM_READ_TIMEOUT_S)
            async with client.stream("POST", url, headers=headers, content=body, timeout=timeout) as resp:
                if resp.status_code != 200:
                    body_text = ""
                    try:
                        body_text = await resp.aread()
                        if isinstance(body_text, (bytes, bytearray)):
                            body_text = body_text.decode("utf-8", "ignore")
                    except Exception:
                        body_text = ""
                    error_message = f"Google AI Studio error: {resp.status_code}"
                    try:
                        error_data = json.loads(body_text) if body_text else {}
                        if isinstance(error_data, dict) and "error" in error_data:
                            error_message = (error_data.get("error") or {}).get("message", error_message)
                    except Exception:
                        pass
                    yield f"data: {json.dumps({'error': {'message': error_message, 'type': 'api_error', 'code': resp.status_code}})}\n\n".encode(
                        "utf-8", "ignore"
                    )
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not isinstance(line, str):
                        line = line.decode("utf-8", "ignore")
                    if not line.startswith("data: "):
                        continue
                    chunk = line[len("data: ") :].strip()
                    if chunk == "[DONE]":
                        return
                    yield f"data: {chunk}\n\n".encode("utf-8", "ignore")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    timeout = httpx.Timeout(timeout=None, connect=UPSTREAM_CONNECT_TIMEOUT_S, read=UPSTREAM_READ_TIMEOUT_S)
    resp = await client.post(url, headers=headers, content=body, timeout=timeout)
    if resp.status_code == 200:
        return Response(
            content=resp.text,
            status_code=200,
            media_type="application/json; charset=utf-8",
        )

    error_message = f"Google AI Studio error: {resp.status_code}"
    try:
        error_data = resp.json()
        if isinstance(error_data, dict) and "error" in error_data:
            error_message = (error_data.get("error") or {}).get("message", error_message)
    except Exception:
        pass
    return Response(
        content=json.dumps({"error": {"message": error_message, "type": "api_error", "code": resp.status_code}}),
        status_code=resp.status_code,
        media_type="application/json",
    )


async def send_gemini_request(payload: dict, is_streaming: bool = False) -> Response:
    """
    Send a request to Google's Gemini API.
    
    Args:
        payload: The request payload in Gemini format
        is_streaming: Whether this is a streaming request
        
    Returns:
        FastAPI Response object
    """
    model_name = payload.get("model")
    quota_block = get_quota_block(model_name)
    if quota_block and (GOOGLE_FALLBACK_TOKEN or "").strip():
        return await _send_ai_studio_request(payload, is_streaming=is_streaming)
    if quota_block:
        now = datetime.now(timezone.utc)
        remaining = quota_block.next_available_at - now
        error_message = (
            "You have exhausted your capacity on this model. "
            f"Your quota will reset after {format_duration(remaining)}."
        )
        headers = {"Retry-After": str(int(max(0, remaining.total_seconds())))}
        return Response(
            content=json.dumps({"error": {"message": error_message, "type": "api_error", "code": 429}}),
            status_code=429,
            media_type="application/json",
            headers=headers,
        )

    # Get and validate credentials
    creds = get_credentials()
    if not creds:
        return Response(
            content="Authentication failed. Please restart the proxy to log in.", 
            status_code=500
        )
    

    # Refresh credentials if needed
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            save_credentials(creds)
        except Exception as e:
            return Response(
                content="Token refresh failed. Please restart the proxy to re-authenticate.", 
                status_code=500
            )
    elif not creds.token:
        return Response(
            content="No access token. Please restart the proxy to re-authenticate.", 
            status_code=500
        )

    # Get project ID and onboard user
    proj_id = await get_user_project_id(creds)
    if not proj_id:
        return Response(content="Failed to get user project ID.", status_code=500)
    
    await onboard_user(creds, proj_id)

    # Build the final payload with project info
    final_payload = {
        "model": payload.get("model"),
        "project": proj_id,
        "request": payload.get("request", {})
    }

    # Determine the action and URL
    action = "streamGenerateContent" if is_streaming else "generateContent"
    target_url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}"
    if is_streaming:
        target_url += "?alt=sse"

    # Build request headers
    request_headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    final_post_data = json.dumps(final_payload)

        # Send the request
    try:
        client = get_http_client()
        retryable_codes = {500, 502, 503, 504}
        retry_cfg = RetryConfig(
            max_attempts=UPSTREAM_MAX_ATTEMPTS,
            base_delay_s=UPSTREAM_BACKOFF_BASE_S,
            max_delay_s=UPSTREAM_BACKOFF_MAX_S,
            retryable_status_codes=frozenset(retryable_codes),
        )

        if is_streaming:
            return _handle_streaming_response(
                client=client,
                target_url=target_url,
                request_headers=request_headers,
                final_post_data=final_post_data,
                retry_cfg=retry_cfg,
                model_name=model_name,
                original_payload=payload,
            )
        timeout = httpx.Timeout(timeout=None, connect=UPSTREAM_CONNECT_TIMEOUT_S, read=UPSTREAM_READ_TIMEOUT_S)
        resp = await post_with_retry(
            client,
            target_url,
            headers=request_headers,
            content=final_post_data,
            timeout=timeout,
            retry_config=retry_cfg,
        )
        if resp.status_code == 429:
            message = None
            try:
                error_data = resp.json()
                if isinstance(error_data, dict):
                    message = (error_data.get("error") or {}).get("message")
            except Exception:
                try:
                    message = resp.text
                except Exception:
                    message = None
            try:
                record_upstream_429(model_name, message, retry_after_s=retry_after_seconds(resp.headers))
            except Exception as e:
                logging.warning(f"Failed to persist upstream quota state: {type(e).__name__}: {e}")
            if (GOOGLE_FALLBACK_TOKEN or "").strip():
                return await _send_ai_studio_request(payload, is_streaming=False)
        return _handle_non_streaming_response(resp)
    except httpx.RequestError as e:
        logging.error(f"Request to Google API failed: {str(e)}")
        return Response(
            content=json.dumps({"error": {"message": f"Request failed: {str(e)}"}}),
            status_code=502,
            media_type="application/json"
        )
    except Exception as e:
        logging.error(f"Unexpected error during Google API request: {str(e)}")
        return Response(
            content=json.dumps({"error": {"message": f"Unexpected error: {str(e)}"}}),
            status_code=500,
            media_type="application/json"
        )


def _handle_streaming_response(
    *,
    client: httpx.AsyncClient,
    target_url: str,
    request_headers: dict,
    final_post_data: str,
    retry_cfg: RetryConfig,
    model_name: Optional[str],
    original_payload: dict,
) -> StreamingResponse:
    """Handle streaming response from Google API."""

    async def stream_generator():
        started = False
        try:
            stream_read_timeout_s = UPSTREAM_STREAM_READ_TIMEOUT_S
            timeout = httpx.Timeout(
                timeout=None,
                connect=UPSTREAM_CONNECT_TIMEOUT_S,
                read=stream_read_timeout_s,
            )

            last_exc: Optional[Exception] = None
            for attempt in range(1, retry_cfg.max_attempts + 1):
                try:
                    async with client.stream(
                        "POST",
                        target_url,
                        headers=request_headers,
                        content=final_post_data,
                        timeout=timeout,
                    ) as resp:
                        if resp.status_code != 200:
                            body_text = ""
                            try:
                                body_text = await resp.aread()
                                if isinstance(body_text, (bytes, bytearray)):
                                    body_text = body_text.decode("utf-8", "ignore")
                            except Exception:
                                body_text = ""

                            logging.error(f"Google API returned status {resp.status_code}: {body_text}")
                            error_message = f"Google API error: {resp.status_code}"
                            try:
                                error_data = json.loads(body_text) if body_text else {}
                                if "error" in error_data:
                                    error_message = error_data["error"].get("message", error_message)
                            except Exception:
                                pass

                            if resp.status_code == 429:
                                try:
                                    record_upstream_429(model_name, error_message, retry_after_s=retry_after_seconds(resp.headers))
                                except Exception as e:
                                    logging.warning(f"Failed to persist upstream quota state: {type(e).__name__}: {e}")
                                if (GOOGLE_FALLBACK_TOKEN or "").strip() and not started:
                                    fallback = await _send_ai_studio_request(original_payload, is_streaming=True)
                                    if isinstance(fallback, StreamingResponse):
                                        async for b in fallback.body_iterator:
                                            yield b
                                    else:
                                        try:
                                            body = fallback.body
                                            if isinstance(body, (bytes, bytearray)):
                                                body = body.decode("utf-8", "ignore")
                                            yield f"data: {body}\n\n".encode("utf-8", "ignore")
                                        except Exception:
                                            yield f"data: {json.dumps({'error': {'message': 'AI Studio fallback failed.', 'type': 'api_error', 'code': 500}})}\n\n".encode('utf-8','ignore')
                                    return

                            if resp.status_code in retry_cfg.retryable_status_codes and attempt < retry_cfg.max_attempts:
                                retry_after_s = retry_after_seconds(resp.headers)
                                await sleep_before_retry(
                                    attempt=attempt + 1,
                                    config=retry_cfg,
                                    retry_after_s=retry_after_s,
                                    reason=f"status={resp.status_code}",
                                )
                                continue

                            error_response = {
                                "error": {
                                    "message": error_message,
                                    "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                                    "code": resp.status_code,
                                }
                            }
                            yield f"data: {json.dumps(error_response)}\n\n".encode("utf-8")
                            return

                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if not isinstance(line, str):
                                line = line.decode("utf-8", "ignore")
                            if not line.startswith("data: "):
                                continue
                            chunk = line[len("data: ") :]
                            try:
                                obj = json.loads(chunk)
                                if isinstance(obj, dict) and "error" in obj:
                                    err = obj.get("error") or {}
                                    if err.get("code") == 429:
                                        try:
                                            record_upstream_429(model_name, err.get("message"))
                                        except Exception as e:
                                            logging.warning(
                                                f"Failed to persist upstream quota state: {type(e).__name__}: {e}"
                                            )
                                if "response" in obj:
                                    response_chunk = obj["response"]
                                    response_json = json.dumps(response_chunk, separators=(",", ":"))
                                    yield f"data: {response_json}\n\n".encode("utf-8", "ignore")
                                    started = True
                                    await asyncio.sleep(0)
                                else:
                                    obj_json = json.dumps(obj, separators=(",", ":"))
                                    yield f"data: {obj_json}\n\n".encode("utf-8", "ignore")
                                    started = True
                            except json.JSONDecodeError:
                                continue
                        return
                except httpx.RequestError as e:
                    last_exc = e
                    if started or attempt >= retry_cfg.max_attempts:
                        raise
                    await sleep_before_retry(
                        attempt=attempt + 1,
                        config=retry_cfg,
                        retry_after_s=None,
                        reason=f"{type(e).__name__}: {e}",
                    )
                    continue
            if last_exc:
                raise last_exc
            raise RuntimeError("Streaming request exhausted without response")
        except httpx.RequestError as e:
            logging.error(f"Streaming request failed: {str(e)}")
            error_response = {
                "error": {
                    "message": f"Upstream request failed: {str(e)}",
                    "type": "api_error",
                    "code": 502,
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n".encode("utf-8", "ignore")
        except Exception as e:
            logging.error(f"Unexpected error during streaming: {str(e)}")
            error_response = {
                "error": {
                    "message": f"An unexpected error occurred: {str(e)}",
                    "type": "api_error",
                    "code": 500
                }
            }
            yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8', "ignore")

    response_headers = {
        "Content-Type": "text/event-stream",
        "Content-Disposition": "attachment",
        "Vary": "Origin, X-Origin, Referer",
        "X-XSS-Protection": "0",
        "X-Frame-Options": "SAMEORIGIN",
        "X-Content-Type-Options": "nosniff",
        "Server": "ESF"
    }
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers=response_headers
    )


def _handle_non_streaming_response(resp: httpx.Response) -> Response:
    """Handle non-streaming response from Google API."""
    if resp.status_code == 200:
        try:
            google_api_response = resp.text
            if google_api_response.startswith('data: '):
                google_api_response = google_api_response[len('data: '):]
            google_api_response = json.loads(google_api_response)
            standard_gemini_response = google_api_response.get("response")
            return Response(
                content=json.dumps(standard_gemini_response),
                status_code=200,
                media_type="application/json; charset=utf-8"
            )
        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Failed to parse Google API response: {str(e)}")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("Content-Type")
            )
    else:
        # Log the error details
        logging.error(f"Google API returned status {resp.status_code}: {resp.text}")
        
        # Try to parse error response and provide meaningful error message
        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", f"API error: {resp.status_code}")
                error_response = {
                    "error": {
                        "message": error_message,
                        "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                        "code": resp.status_code
                    }
                }
                return Response(
                    content=json.dumps(error_response),
                    status_code=resp.status_code,
                    media_type="application/json"
                )
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback to original response if we can't parse the error
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("Content-Type")
        )


def build_gemini_payload_from_openai(openai_payload: dict) -> dict:
    """
    Build a Gemini API payload from an OpenAI-transformed request.
    This is used when OpenAI requests are converted to Gemini format.
    """
    # Extract model from the payload
    model = openai_payload.get("model")
    
    # Get safety settings or use defaults
    safety_settings = openai_payload.get("safetySettings", DEFAULT_SAFETY_SETTINGS)
    
    # Build the request portion
    request_data = {
        "contents": openai_payload.get("contents"),
        "systemInstruction": openai_payload.get("systemInstruction"),
        "cachedContent": openai_payload.get("cachedContent"),
        "tools": openai_payload.get("tools"),
        "toolConfig": openai_payload.get("toolConfig"),
        "safetySettings": safety_settings,
        "generationConfig": openai_payload.get("generationConfig", {}),
    }
    
    # Remove any keys with None values
    request_data = {k: v for k, v in request_data.items() if v is not None}
    
    return {
        "model": model,
        "request": request_data
    }


def build_gemini_payload_from_native(native_request: dict, model_from_path: str) -> dict:
    """
    Build a Gemini API payload from a native Gemini request.
    This is used for direct Gemini API calls.
    """
    native_request["safetySettings"] = DEFAULT_SAFETY_SETTINGS
    
    if "generationConfig" not in native_request:
        native_request["generationConfig"] = {}
        
    # native_request["enableEnhancedCivicAnswers"] = False
    
    if "thinkingConfig" not in native_request["generationConfig"]:
        native_request["generationConfig"]["thinkingConfig"] = {}
    
    if "gemini-2.5-flash-image" not in model_from_path:
        # Configure thinking based on model variant
        thinking_budget = get_thinking_budget(model_from_path)
        include_thoughts = should_include_thoughts(model_from_path)
    
        native_request["generationConfig"]["thinkingConfig"]["includeThoughts"] = include_thoughts
        if "thinkingBudget" in native_request["generationConfig"]["thinkingConfig"]:
            pass
        else:
            native_request["generationConfig"]["thinkingConfig"]["thinkingBudget"] = thinking_budget
    
    # Add Google Search grounding for search models
    if is_search_model(model_from_path):
        if "tools" not in native_request:
            native_request["tools"] = []
        # Add googleSearch tool if not already present
        if not any(tool.get("googleSearch") for tool in native_request["tools"]):
            native_request["tools"].append({"googleSearch": {}})
    
    return {
        "model": get_base_model_name(model_from_path),  # Use base model name for API call
        "request": native_request
    }
