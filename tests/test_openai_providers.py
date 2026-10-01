"""OpenAI provider code against a mocked HTTP transport: request shapes + response parsing, no key needed."""

import base64
import io
import json
import wave

import httpx2 as httpx

from reelgen.config import load_settings
from reelgen.models import ParsedInput
from reelgen.providers.openai_impl import OpenAIImages, OpenAIJudge, OpenAISTT, OpenAITTS, OpenAIWriter

PLAN = {
    "title": "Storm",
    "logline": "A keeper refuses",
    "visual_style": "moody oil painting",
    "characters": [{"name": "Keeper", "appearance": "old man, grey beard", "voice": "male_deep"}],
    "shots": [
        {
            "shot_id": 1,
            "lines": [{"line_index": 0, "speaker": "Keeper", "delivery": "hoarse whisper, stubborn"}],
            "on_screen": ["Keeper"],
            "visual_prompt": "Old keeper with grey beard in a lighthouse window, storm, ship lights",
            "camera": "zoom_in",
        }
    ],
}
REVIEW = {"score": 4, "has_text_or_watermark": False, "has_anatomy_defects": False, "issues": []}


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 2400)
    return buf.getvalue()


def _response(payload: dict) -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "m",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(payload), "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def make_client(seen: list):
    from openai import OpenAI

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, req.content))
        path = req.url.path
        if path.endswith("/responses"):
            body = json.loads(req.content)
            is_judge = isinstance(body["input"], list)
            return httpx.Response(200, json=_response(REVIEW if is_judge else PLAN))
        if path.endswith("/audio/speech"):
            return httpx.Response(200, content=_wav_bytes(), headers={"content-type": "audio/wav"})
        if path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "Not tonight."})
        if path.endswith("/images/generations") or path.endswith("/images/edits"):
            png = base64.b64encode(b"\x89PNG fake").decode()
            return httpx.Response(200, json={"created": 0, "data": [{"b64_json": png}]})
        return httpx.Response(404, json={"error": {"message": path}})

    return OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_openai_providers_roundtrip(tmp_path):
    seen: list = []
    client, s, k = make_client(seen), load_settings(demo=False), {}
    parsed = ParsedInput(idea="i", lines=["Not tonight."], lines_source="quotes", lang="en", est_speech_s=1)

    plan = OpenAIWriter(client, s, k).write(parsed, feedback=["fix this"])
    assert plan.shots[0].lines[0].line_index == 0
    sent = json.loads(seen[0][1])
    assert sent["text"]["format"]["type"] == "json_schema" and "[0] Not tonight." in sent["input"]
    assert "fix this" in sent["input"]

    wav = tmp_path / "l.wav"
    OpenAITTS(client, s, k).synthesize("Not tonight.", "male_deep", "hoarse whisper", "en", wav, attempt=1)
    speech = json.loads(seen[1][1])
    assert speech["input"] == "Not tonight." and speech["voice"] == "onyx" and speech["speed"] < 1
    assert "hoarse whisper" in speech["instructions"]  # scene emotion reaches the voice actor
    assert wav.exists()

    assert OpenAISTT(client, s, k).transcribe(wav, "en") == "Not tonight."

    png = tmp_path / "s.png"
    OpenAIImages(client, s, k).generate("prompt", png, 0)
    assert png.read_bytes().startswith(b"\x89PNG")

    assert OpenAIJudge(client, s, k).review(png, "prompt").score == 4

    # with character references -> images/edits (multipart with the reference file)
    OpenAIImages(client, s, k).generate("shot with keeper", tmp_path / "s2.png", 0, refs=[png])
    path, body = seen[-1]
    assert path.endswith("/images/edits") and b"shot with keeper" in body and b"PNG fake" in body
    assert k["llm_calls"] == 1 and k["tts_calls"] == 1 and k["image_calls"] == 2 and k["judge_calls"] == 1
