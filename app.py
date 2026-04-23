import os
import time
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-E4B-it")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

TARGET_SR = 16000
MAX_CHUNK_SECONDS = float(os.getenv("MAX_CHUNK_SECONDS", "29.5"))
CHUNK_OVERLAP_SECONDS = float(os.getenv("CHUNK_OVERLAP_SECONDS", "1.0"))
MIN_CHUNK_SECONDS = float(os.getenv("MIN_CHUNK_SECONDS", "3.0"))
PAD_TO_MIN_CHUNK = os.getenv("PAD_TO_MIN_CHUNK", "true").lower() in {"1", "true", "yes", "on"}

MAX_PREPROCESS_WORKERS = int(os.getenv("MAX_PREPROCESS_WORKERS", str(max(2, (os.cpu_count() or 4) // 2))))
MAX_PARALLEL_CHUNKS = int(os.getenv("MAX_PARALLEL_CHUNKS", "2"))
MAX_PARALLEL_REQUESTS = int(os.getenv("MAX_PARALLEL_REQUESTS", "2"))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
AUTH_BEARER = os.getenv("AUTH_BEARER", "").strip()
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="OpenAI-compatible Audio API (Gemma 4 E4B)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # mieux: liste explicite en prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


processor = None
model = None

executor = ThreadPoolExecutor(max_workers=MAX_PREPROCESS_WORKERS)
gpu_chunk_semaphore = asyncio.Semaphore(MAX_PARALLEL_CHUNKS)
request_semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)


def check_auth(authorization: Optional[str]) -> None:
    if not AUTH_BEARER:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization[7:].strip()
    if token != AUTH_BEARER:
        raise HTTPException(status_code=401, detail="Unauthorized")


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def dedupe_overlap_text(prev_text: str, curr_text: str) -> str:
    a = normalize_text(prev_text)
    b = normalize_text(curr_text)

    if not a:
        return b
    if not b:
        return ""

    a_words = a.split()
    b_words = b.split()
    max_overlap = min(len(a_words), len(b_words), 32)

    overlap_len = 0
    for k in range(max_overlap, 0, -1):
        if a_words[-k:] == b_words[:k]:
            overlap_len = k
            break

    if overlap_len > 0:
        b_words = b_words[overlap_len:]

    return " ".join(b_words).strip()


def sec_to_srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def sec_to_vtt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_prompt(language: Optional[str], prompt: Optional[str], translate: bool = False) -> str:
    if translate:
        if language:
            base = (
                f"Transcribe the following speech segment, then translate it into {language}. "
                "Only output the final translation."
            )
        else:
            base = (
                "Transcribe the following speech segment, then translate it into English. "
                "Only output the final translation."
            )
    else:
        if language:
            base = (
                f"Transcribe the following speech segment in {language} into {language} text.\n"
                "Follow these specific instructions for formatting the answer:\n"
                "* Only output the transcription, with no newlines.\n"
                "* When transcribing numbers, write the digits."
            )
        else:
            base = (
                "Transcribe the following speech segment in its original language.\n"
                "Follow these specific instructions for formatting the answer:\n"
                "* Only output the transcription, with no newlines.\n"
                "* When transcribing numbers, write the digits."
            )

    if prompt:
        base += f"\nAdditional context: {prompt.strip()}"

    return base


def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "audio.bin").suffix or ".bin"
    fd, path = tempfile.mkstemp(prefix="audio_", suffix=suffix)
    os.close(fd)

    with open(path, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def ffmpeg_decode_to_float32_mono_16k(input_path: str) -> Tuple[np.ndarray, int, float]:
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel", "error",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg decode failed: {err.strip()}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)

    if audio.size == 0:
        duration = 0.0
    else:
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
        duration = audio.shape[0] / float(TARGET_SR)

    return audio, TARGET_SR, duration


def make_chunks(duration: float) -> List[Tuple[float, float]]:
    if duration <= 0:
        return []

    if duration <= MAX_CHUNK_SECONDS:
        return [(0.0, duration)]

    chunks: List[Tuple[float, float]] = []
    step = max(0.25, MAX_CHUNK_SECONDS - CHUNK_OVERLAP_SECONDS)

    start = 0.0
    while start < duration:
        end = min(duration, start + MAX_CHUNK_SECONDS)
        chunks.append((start, end))
        if end >= duration:
            break
        start += step

    if len(chunks) >= 2:
        last_start, last_end = chunks[-1]
        if (last_end - last_start) < MIN_CHUNK_SECONDS:
            prev_start, _ = chunks[-2]
            chunks[-2] = (prev_start, last_end)
            chunks.pop()

    return [(round(s, 6), round(e, 6)) for s, e in chunks if e > s]


def slice_audio(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start = max(0, int(round(start_sec * sr)))
    end = min(audio.shape[0], int(round(end_sec * sr)))
    if end <= start:
        return np.zeros((0,), dtype=np.float32)

    chunk = audio[start:end].astype(np.float32, copy=False)

    min_samples = int(round(MIN_CHUNK_SECONDS * sr))
    if PAD_TO_MIN_CHUNK and chunk.shape[0] < min_samples:
        pad = np.zeros((min_samples - chunk.shape[0],), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=0)

    return np.clip(chunk, -1.0, 1.0).astype(np.float32, copy=False)


def write_chunk_wav_float32(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio, sr, subtype="FLOAT", format="WAV")


def load_model_once() -> None:
    global processor, model

    if processor is not None and model is not None:
        return

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    local_model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    local_processor = AutoProcessor.from_pretrained(MODEL_ID)

    try:
        local_model.eval()
    except Exception:
        pass

    model = local_model
    processor = local_processor


def transcribe_one_chunk_sync(chunk_wav_path: str, instruction: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": chunk_wav_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    try:
        inputs = inputs.to(model.device)
    except Exception:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated = outputs[0][input_len:]
    response = processor.decode(generated, skip_special_tokens=False)

    try:
        parsed = processor.parse_response(response)
        if isinstance(parsed, dict):
            if parsed.get("text"):
                return normalize_text(parsed["text"])
            if parsed.get("response"):
                return normalize_text(parsed["response"])
        elif isinstance(parsed, str):
            return normalize_text(parsed)
    except Exception:
        pass

    cleaned = response
    for marker in [
        "<bos>",
        "<eos>",
        "<|turn>model",
        "<|turn>user",
        "<|audio|>",
        "<audio|>",
        "<turn|>",
    ]:
        cleaned = cleaned.replace(marker, " ")

    return normalize_text(cleaned)


async def transcribe_one_chunk_async(chunk_audio: np.ndarray, sr: int, instruction: str) -> str:
    fd, path = tempfile.mkstemp(prefix="chunk_", suffix=".wav")
    os.close(fd)

    try:
        await asyncio.to_thread(write_chunk_wav_float32, path, chunk_audio, sr)
        async with gpu_chunk_semaphore:
            return await asyncio.to_thread(transcribe_one_chunk_sync, path, instruction)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


async def preprocess_audio(upload: UploadFile) -> Tuple[np.ndarray, int, float]:
    temp_path = save_upload_to_temp(upload)
    try:
        return await asyncio.get_running_loop().run_in_executor(
            executor,
            ffmpeg_decode_to_float32_mono_16k,
            temp_path,
        )
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def build_verbose_json(
    full_text: str,
    duration: float,
    language: Optional[str],
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "task": "transcribe",
        "language": language or "unknown",
        "duration": duration,
        "text": full_text,
        "segments": segments,
    }


def render_srt(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{sec_to_srt_time(seg['start'])} --> {sec_to_srt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_vtt(segments: List[Dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{sec_to_vtt_time(seg['start'])} --> {sec_to_vtt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


async def transcribe_audio_array(
    audio: np.ndarray,
    sr: int,
    duration: float,
    language: Optional[str],
    prompt: Optional[str],
    translate: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    instruction = build_prompt(language=language, prompt=prompt, translate=translate)
    chunks = make_chunks(duration)

    if not chunks:
        return "", []

    loop = asyncio.get_running_loop()
    slice_tasks = [
        loop.run_in_executor(executor, slice_audio, audio, sr, start, end)
        for (start, end) in chunks
    ]
    chunk_arrays = await asyncio.gather(*slice_tasks)

    text_tasks = [
        transcribe_one_chunk_async(chunk_audio, sr, instruction)
        for chunk_audio in chunk_arrays
    ]
    chunk_texts = await asyncio.gather(*text_tasks)

    merged_parts: List[str] = []
    segments: List[Dict[str, Any]] = []
    prev = ""

    for ((start, end), text) in zip(chunks, chunk_texts):
        text = normalize_text(text)
        if not text:
            continue

        cleaned = text if not prev else dedupe_overlap_text(prev, text)
        if cleaned:
            merged_parts.append(cleaned)
            prev = normalize_text(prev + " " + cleaned)
            segments.append({
                "id": len(segments),
                "seek": int(round(start * 1000)),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": cleaned,
            })

    return normalize_text(" ".join(merged_parts)), segments


@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(load_model_once)


@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=False, cancel_futures=True)


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_ID}


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": "whisper-1", "object": "model", "created": now, "owned_by": "local"},
            {"id": "gpt-4o-mini-transcribe", "object": "model", "created": now, "owned_by": "local"},
            {"id": "gemma-4-e4b-it", "object": "model", "created": now, "owned_by": "google"},
            {"id": MODEL_ID, "object": "model", "created": now, "owned_by": "google"},
        ],
    }


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model_form: str = Form(..., alias="model"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    temperature: Optional[float] = Form(default=None),
    timestamp_granularities: Optional[List[str]] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    check_auth(authorization)
    _ = model_form
    _ = temperature
    _ = timestamp_granularities

    async with request_semaphore:
        try:
            audio, sr, duration = await preprocess_audio(file)
            if sr != TARGET_SR:
                raise RuntimeError(f"unexpected sample rate after preprocessing: {sr}")

            text, segments = await transcribe_audio_array(
                audio=audio,
                sr=sr,
                duration=duration,
                language=language,
                prompt=prompt,
                translate=False,
            )

            fmt = (response_format or "json").lower()
            if fmt == "text":
                return PlainTextResponse(text)
            if fmt == "json":
                return JSONResponse({"text": text})
            if fmt == "verbose_json":
                return JSONResponse(build_verbose_json(text, duration, language, segments))
            if fmt == "srt":
                return Response(render_srt(segments), media_type="text/plain; charset=utf-8")
            if fmt == "vtt":
                return Response(render_vtt(segments), media_type="text/vtt; charset=utf-8")

            raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ASR pipeline error: {e}")


@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model_form: str = Form(..., alias="model"),
    prompt: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    temperature: Optional[float] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    check_auth(authorization)
    _ = model_form
    _ = temperature

    async with request_semaphore:
        try:
            audio, sr, duration = await preprocess_audio(file)
            if sr != TARGET_SR:
                raise RuntimeError(f"unexpected sample rate after preprocessing: {sr}")

            text, segments = await transcribe_audio_array(
                audio=audio,
                sr=sr,
                duration=duration,
                language="English",
                prompt=prompt,
                translate=True,
            )

            fmt = (response_format or "json").lower()
            if fmt == "text":
                return PlainTextResponse(text)
            if fmt == "json":
                return JSONResponse({"text": text})
            if fmt == "verbose_json":
                return JSONResponse(build_verbose_json(text, duration, "english", segments))
            if fmt == "srt":
                return Response(render_srt(segments), media_type="text/plain; charset=utf-8")
            if fmt == "vtt":
                return Response(render_vtt(segments), media_type="text/vtt; charset=utf-8")

            raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ASR pipeline error: {e}")
