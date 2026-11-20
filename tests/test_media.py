import base64
from pathlib import Path

import pytest

from omnillm.core.errors import InvalidRequestError
from omnillm.core.media import (
    encode_image_to_base64,
    encode_image_to_data_uri,
    has_images,
    normalize_messages_for_ollama,
    normalize_messages_for_openai,
    sanitize_untrusted_messages,
)


def test_encode_image_to_base64(tmp_path: Path):
    raw = b"sample_pixels"
    expected = base64.b64encode(raw).decode("ascii")

    # From raw bytes
    assert encode_image_to_base64(raw) == expected

    # From Path
    file_path = tmp_path / "test.png"
    file_path.write_bytes(raw)
    assert encode_image_to_base64(file_path) == expected

    # From str path
    assert encode_image_to_base64(str(file_path)) == expected

    # From data URI
    data_uri = f"data:image/png;base64,{expected}"
    assert encode_image_to_base64(data_uri) == expected

    # Already base64 string
    assert encode_image_to_base64(expected) == expected

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported image type"):
        encode_image_to_base64(12345)  # type: ignore[arg-type]


def test_encode_image_to_data_uri(tmp_path: Path):
    raw = b"png_data"
    b64 = base64.b64encode(raw).decode("ascii")

    # Already data uri
    uri = f"data:image/webp;base64,{b64}"
    assert encode_image_to_data_uri(uri) == uri

    # Path detection
    png_file = tmp_path / "img.png"
    png_file.write_bytes(raw)
    assert encode_image_to_data_uri(png_file) == f"data:image/png;base64,{b64}"

    # Default fallback
    assert encode_image_to_data_uri(raw) == f"data:image/jpeg;base64,{b64}"


def test_has_images():
    assert not has_images([{"role": "user", "content": "Just text"}])
    assert has_images([{"role": "user", "content": "Prompt", "images": ["data:image/png;base64,abc"]}])
    assert has_images(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Prompt"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
    )


def test_normalize_messages_for_ollama(tmp_path: Path):
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(b"image_content")
    b64 = base64.b64encode(b"image_content").decode("ascii")

    # OpenAI format converted to Ollama
    openai_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": str(img_file)}},
            ],
        }
    ]
    ollama_msgs = normalize_messages_for_ollama(openai_messages)
    assert ollama_msgs == [{"role": "user", "content": "What is this?", "images": [b64]}]

    # Native Ollama format with images field
    native_messages = [{"role": "user", "content": "Describe", "images": [img_file]}]
    assert normalize_messages_for_ollama(native_messages) == [{"role": "user", "content": "Describe", "images": [b64]}]


def test_normalize_messages_for_openai(tmp_path: Path):
    img_file = tmp_path / "photo.png"
    img_file.write_bytes(b"pixel_data")
    b64 = base64.b64encode(b"pixel_data").decode("ascii")

    # Native Ollama images converted to OpenAI format
    ollama_messages = [{"role": "user", "content": "Describe", "images": [img_file]}]
    openai_msgs = normalize_messages_for_openai(ollama_messages)

    assert openai_msgs == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ]


def test_sanitize_never_reads_a_base64_shaped_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    # "abcd" is valid base64 *and* an existing relative path: the server must treat it as data, never read the file.
    (tmp_path / "abcd").write_bytes(b"top secret")

    result = sanitize_untrusted_messages([{"role": "user", "content": "x", "images": ["abcd"]}])

    assert result[0]["images"] == ["data:image/jpeg;base64,abcd"]
    ollama_messages = normalize_messages_for_ollama(result)
    assert ollama_messages[0]["images"] == ["abcd"]
    openai_messages = normalize_messages_for_openai(result)
    assert openai_messages[0]["content"][-1]["image_url"]["url"] == "data:image/jpeg;base64,abcd"


@pytest.mark.parametrize(
    "image",
    [
        "https://example.com/cat.png",
        "data:text/plain;base64,aGVsbG8=",
        "data:image/png,rawdata",
        "data:image/png;base64,***",
        "",
    ],
)
def test_sanitize_rejects_remote_urls_and_malformed_images(image: str):
    with pytest.raises(InvalidRequestError):
        sanitize_untrusted_messages([{"role": "user", "content": "x", "images": [image]}])
    with pytest.raises(InvalidRequestError):
        sanitize_untrusted_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}])


def test_sanitize_normalizes_image_url_parts_and_keeps_other_fields():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8=", "detail": "low"}},
            ],
        }
    ]

    result = sanitize_untrusted_messages(messages)

    assert result[0]["content"][0] == {"type": "text", "text": "hi"}
    assert result[0]["content"][1]["image_url"] == {"url": "data:image/png;base64,aGVsbG8=", "detail": "low"}
    assert messages[0]["content"][1]["image_url"]["detail"] == "low"  # input not mutated


def test_tool_calls_are_converted_between_openai_and_ollama_shapes():
    openai_call = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
    history = [
        {"role": "assistant", "content": "", "tool_calls": [openai_call]},
        {"role": "tool", "tool_call_id": "call_1", "name": "f", "content": "2"},
    ]

    ollama = normalize_messages_for_ollama(history)
    assert ollama[0]["tool_calls"] == [{"function": {"name": "f", "arguments": {"a": 1}}}]
    assert ollama[1]["tool_name"] == "f"

    ollama_history = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}
    ]
    openai = normalize_messages_for_openai(ollama_history)
    call = openai[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"] == {"name": "f", "arguments": '{"a": 1}'}
