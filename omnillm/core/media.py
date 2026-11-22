import base64
import binascii
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omnillm.core.errors import InvalidImageError
from omnillm.core.tool_calls import to_ollama_tool_call, to_openai_tool_call
from omnillm.core.types import ChatMessage


def encode_image_to_base64(image: str | bytes | Path) -> str:
    """Encodes an image input (file path, Path, bytes, data URI, or raw base64) into a pure base64 string."""
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii")
    if isinstance(image, Path):
        return base64.b64encode(image.read_bytes()).decode("ascii")
    if isinstance(image, str):
        if image.startswith("data:image/") and ";base64," in image:
            return image.split(";base64,", 1)[1]
        path = Path(image)
        if path.is_file():
            return base64.b64encode(path.read_bytes()).decode("ascii")
        # Already a base64 string
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


def encode_image_to_data_uri(image: str | bytes | Path, default_mime: str = "image/jpeg") -> str:
    """Converts an image input into an RFC 2397 data URI string (e.g. data:image/jpeg;base64,...)."""
    if isinstance(image, str) and image.startswith("data:image/"):
        return image

    mime = default_mime
    if isinstance(image, Path) or (isinstance(image, str) and Path(image).is_file()):
        p = Path(image)
        suffix = p.suffix.lower().lstrip(".")
        if suffix in {"png", "webp", "gif"}:
            mime = f"image/{suffix}"
        elif suffix in {"jpg", "jpeg"}:
            mime = "image/jpeg"

    b64 = encode_image_to_base64(image)
    return f"data:{mime};base64,{b64}"


def has_images(messages: Sequence[ChatMessage]) -> bool:
    """Checks whether any message in the sequence contains image payloads or image_url blocks."""
    for msg in messages:
        if msg.get("images"):
            return True
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def normalize_messages_for_ollama(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Normalizes messages for Ollama, which expects string content and an optional 'images' list of base64 strings."""
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        msg_dict = dict(msg)
        content = msg_dict.get("content")
        raw_images = msg_dict.pop("images", None) or []
        images: list[str] = [encode_image_to_base64(img) for img in raw_images]

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        text_parts.append(part.get("text") or "")
                    elif part_type == "image_url":
                        img_info = part.get("image_url")
                        url = img_info.get("url") if isinstance(img_info, dict) else img_info
                        if url:
                            images.append(encode_image_to_base64(url))
            msg_dict["content"] = "\n".join(text_parts)
        elif content is None:
            msg_dict["content"] = ""

        if images:
            msg_dict["images"] = images

        if msg_dict.get("tool_calls"):
            msg_dict["tool_calls"] = [to_ollama_tool_call(call) for call in msg_dict["tool_calls"]]
        if msg_dict.get("role") == "tool" and msg_dict.get("name") and not msg_dict.get("tool_name"):
            msg_dict["tool_name"] = msg_dict["name"]

        normalized.append(msg_dict)
    return normalized


def normalize_messages_for_openai(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Normalizes messages for OpenAI-compatible backends (like llama.cpp), which expect content parts with image_url."""
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        msg_dict = dict(msg)
        content = msg_dict.get("content")
        raw_images = msg_dict.pop("images", None) or []

        if raw_images:
            parts: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                parts.extend(content)
            for img in raw_images:
                parts.append({"type": "image_url", "image_url": {"url": encode_image_to_data_uri(img)}})
            msg_dict["content"] = parts
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    img_info = part.get("image_url")
                    url = img_info.get("url") if isinstance(img_info, dict) else img_info
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": encode_image_to_data_uri(url)}})
                else:
                    parts.append(part)
            msg_dict["content"] = parts

        if msg_dict.get("tool_calls"):
            msg_dict["tool_calls"] = [to_openai_tool_call(call) for call in msg_dict["tool_calls"]]

        normalized.append(msg_dict)
    return normalized


def _untrusted_image_to_data_uri(image: Any) -> str:
    """Validates an image supplied by a remote client and returns it as a data URI.

    Only inline data is accepted. Local file paths and remote URLs are rejected so that an API client can
    never make the server read files from its own disk or fetch arbitrary URLs.
    """
    if not isinstance(image, str) or not image:
        raise InvalidImageError("Images must be base64 strings or data:image/...;base64,... URIs.")
    if image.startswith(("http://", "https://")):
        raise InvalidImageError("Remote image URLs are not supported; send the image inline as a base64 data URI.")
    if image.startswith("data:"):
        header, sep, payload = image.partition(",")
        if not sep or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise InvalidImageError("Image data URIs must have the form data:image/<type>;base64,<data>.")
        mime = header[len("data:") : -len(";base64")]
    else:
        mime, payload = "image/jpeg", image
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise InvalidImageError("Image payload is not valid base64.") from error
    if not decoded:
        raise InvalidImageError("Image payload is empty.")
    return f"data:{mime};base64,{base64.b64encode(decoded).decode('ascii')}"


def sanitize_untrusted_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Rewrites every image in messages from an untrusted source (e.g. HTTP clients) into a validated data URI.

    Data URIs are never interpreted as file paths by the encoders above, which closes the local-file read path.
    """
    sanitized: list[dict[str, Any]] = []
    for msg in messages:
        msg_dict = dict(msg)
        if msg_dict.get("images"):
            msg_dict["images"] = [_untrusted_image_to_data_uri(img) for img in msg_dict["images"]]
        content = msg_dict.get("content")
        if isinstance(content, list):
            parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    img_info = part.get("image_url")
                    url = img_info.get("url") if isinstance(img_info, dict) else img_info
                    new_info = dict(img_info) if isinstance(img_info, dict) else {}
                    new_info["url"] = _untrusted_image_to_data_uri(url)
                    part = {**part, "image_url": new_info}
                parts.append(part)
            msg_dict["content"] = parts
        sanitized.append(msg_dict)
    return sanitized
