"""
FastAPI server for streaming Chatterbox TTS
Streams audio chunks back to client as they're generated
"""
import io
import base64
import os
import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from huggingface_hub import login

from chatterbox.tts import ChatterboxTTS

# Login to HuggingFace if token is available
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✓ Logged in to Hugging Face")

# Initialize FastAPI
app = FastAPI(title="Chatterbox Streaming TTS API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
MODEL = None
DEVICE = None


class TTSRequest(BaseModel):
    text: str
    chunk_size: int = 25
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    audio_prompt_base64: Optional[str] = None


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global MODEL, DEVICE

    # Auto-detect device
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    print(f"Loading Chatterbox model on {DEVICE}...")
    MODEL = ChatterboxTTS.from_pretrained(device=DEVICE)
    print(f"✓ Model loaded successfully on {DEVICE}")


@app.get("/")
async def root():
    return {
        "service": "Chatterbox Streaming TTS",
        "device": DEVICE,
        "endpoints": {
            "POST /stream/audio": "Stream audio chunks as raw PCM",
            "POST /stream/wav": "Stream complete WAV file",
            "POST /stream/chunks": "Stream base64-encoded audio chunks (JSON)",
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": DEVICE
    }


def audio_chunk_generator(
    text: str,
    chunk_size: int,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    audio_prompt_path: Optional[str] = None
):
    """Generator that yields raw audio chunks"""
    try:
        kwargs = {
            "text": text,
            "chunk_size": chunk_size,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "print_metrics": False
        }

        if audio_prompt_path:
            kwargs["audio_prompt_path"] = audio_prompt_path

        for audio_chunk, metrics in MODEL.generate_stream(**kwargs):
            # Convert to bytes (PCM s16le format)
            audio_np = audio_chunk.squeeze().numpy()
            # Scale to int16 range
            audio_int16 = (audio_np * 32767).astype('int16')
            yield audio_int16.tobytes()

    except Exception as e:
        print(f"Error in audio generation: {e}")
        raise


@app.post("/stream/audio")
async def stream_audio(request: TTSRequest):
    """
    Stream raw PCM audio chunks in real-time
    Returns: audio/pcm stream (s16le, 24kHz, mono)

    Usage with curl:
    curl -X POST http://localhost:8000/stream/audio \
      -H "Content-Type: application/json" \
      -d '{"text": "Hello world"}' \
      --output output.raw

    Play with: ffplay -f s16le -ar 24000 -ac 1 output.raw
    """
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return StreamingResponse(
        audio_chunk_generator(
            text=request.text,
            chunk_size=request.chunk_size,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
        ),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
            "X-Sample-Format": "s16le"
        }
    )


@app.post("/stream/wav")
async def stream_wav(request: TTSRequest):
    """
    Generate complete audio and stream as WAV file
    Returns: audio/wav stream

    Usage with curl:
    curl -X POST http://localhost:8000/stream/wav \
      -H "Content-Type: application/json" \
      -d '{"text": "Hello world"}' \
      --output output.wav
    """
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Collect all chunks
    audio_chunks = []
    for audio_chunk, _ in MODEL.generate_stream(
        text=request.text,
        chunk_size=request.chunk_size,
        exaggeration=request.exaggeration,
        cfg_weight=request.cfg_weight,
        temperature=request.temperature,
        print_metrics=False
    ):
        audio_chunks.append(audio_chunk)

    # Combine and save to buffer
    final_audio = torch.cat(audio_chunks, dim=-1)

    # Create WAV in memory
    buffer = io.BytesIO()
    ta.save(buffer, final_audio, MODEL.sr, format="wav")
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.read()]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"}
    )


@app.post("/stream/chunks")
async def stream_chunks_json(request: TTSRequest):
    """
    Stream audio chunks as JSON with base64-encoded audio
    Returns: text/event-stream (Server-Sent Events)

    Each event contains:
    {
        "chunk_id": 1,
        "audio_base64": "...",
        "duration": 1.0,
        "sample_rate": 24000
    }

    Usage with curl:
    curl -X POST http://localhost:8000/stream/chunks \
      -H "Content-Type: application/json" \
      -d '{"text": "Hello world"}' \
      --no-buffer
    """
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def event_generator():
        try:
            chunk_id = 0
            for audio_chunk, metrics in MODEL.generate_stream(
                text=request.text,
                chunk_size=request.chunk_size,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                print_metrics=False
            ):
                chunk_id += 1

                # Convert to base64
                audio_np = audio_chunk.squeeze().numpy()
                audio_int16 = (audio_np * 32767).astype('int16')
                audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')

                duration = audio_chunk.shape[-1] / MODEL.sr

                # SSE format
                event_data = {
                    "chunk_id": chunk_id,
                    "audio_base64": audio_b64,
                    "duration": duration,
                    "sample_rate": MODEL.sr,
                    "latency_to_first_chunk": metrics.latency_to_first_chunk,
                }

                yield f"data: {event_data}\n\n"

            # Send completion event
            yield f"data: {{'status': 'completed', 'total_chunks': {chunk_id}}}\n\n"

        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/generate")
async def generate_complete(request: TTSRequest):
    """
    Non-streaming endpoint - returns complete audio as base64
    Useful for simple integrations
    """
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Collect all chunks
    audio_chunks = []
    total_time = 0

    for audio_chunk, metrics in MODEL.generate_stream(
        text=request.text,
        chunk_size=request.chunk_size,
        exaggeration=request.exaggeration,
        cfg_weight=request.cfg_weight,
        temperature=request.temperature,
        print_metrics=False
    ):
        audio_chunks.append(audio_chunk)
        if metrics.total_generation_time:
            total_time = metrics.total_generation_time

    # Combine audio
    final_audio = torch.cat(audio_chunks, dim=-1)

    # Convert to base64
    buffer = io.BytesIO()
    ta.save(buffer, final_audio, MODEL.sr, format="wav")
    buffer.seek(0)
    audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {
        "audio_base64": audio_b64,
        "sample_rate": MODEL.sr,
        "duration": final_audio.shape[-1] / MODEL.sr,
        "generation_time": total_time,
        "format": "wav"
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    uvicorn.run(
        "api_streaming:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
        log_level="info"
    )