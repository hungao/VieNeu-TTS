"""
FastAPI server for VieNeu-TTS
Provides REST API endpoints for text-to-speech synthesis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import base64
import soundfile as sf
import tempfile
import yaml
import os
import sys
from typing import Optional
import numpy as np
from vieneu_tts import VieNeuTTS, FastVieNeuTTS
import torch
from utils.core_utils import split_text_into_chunks
from functools import lru_cache

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Cannot read config.yaml: {e}")

VOICE_SAMPLES = _config.get("voice_samples", {})
BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})
_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)

# Global TTS instance
tts_instance = None
current_config = {
    "backbone": None,
    "codec": None,
    "device": None,
    "use_lmdeploy": False
}

# Cache for reference texts
@lru_cache(maxsize=32)
def get_ref_text_cached(text_path: str) -> str:
    """Cache reference text loading"""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()

# Pydantic models for request/response
class SynthesizeRequest(BaseModel):
    text: str
    voice: str
    mode: Optional[str] = "standard"  # "standard" or "streaming"

class BatchSynthesizeRequest(BaseModel):
    texts: list[str]
    voice: str

class SynthesizeResponse(BaseModel):
    audio: str  # base64 encoded WAV
    duration: float  # in seconds

class BatchSynthesizeResponse(BaseModel):
    results: list[dict]
    count: int
    totalDuration: float

class VoiceInfo(BaseModel):
    name: str
    audio_path: str
    text_path: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str
    backbone: Optional[str]
    codec: Optional[str]

class ModelConfigRequest(BaseModel):
    backbone: str = "VieNeu-TTS-0.3B (GPU)"
    codec: str = "NeuCodec (Distill)"
    device: str = "Auto"
    use_lmdeploy: bool = True

# Create FastAPI app
app = FastAPI(
    title="VieNeu-TTS API",
    description="REST API for Vietnamese text-to-speech with voice cloning",
    version="1.0.0"
)

def should_use_lmdeploy(device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend"""
    if sys.platform == "darwin":
        return False

    if device_choice == "Auto":
        has_gpu = torch.cuda.is_available()
    elif device_choice == "CUDA":
        has_gpu = torch.cuda.is_available()
    else:
        has_gpu = False

    return has_gpu

def get_tts_instance():
    """Get or create TTS instance"""
    global tts_instance
    if tts_instance is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded. Please call POST /api/load-model first."
        )
    return tts_instance

def encode_audio_to_base64(audio_array: np.ndarray, sample_rate: int = 24000) -> str:
    """Encode audio array to base64 WAV"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio_array, sample_rate)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()

    os.unlink(tmp_path)
    return base64.b64encode(audio_bytes).decode('utf-8')

def get_voice_info(voice_name: str) -> tuple:
    """Get voice audio path and reference text"""
    if voice_name not in VOICE_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice_name}' not found. Available voices: {list(VOICE_SAMPLES.keys())}"
        )

    voice_info = VOICE_SAMPLES[voice_name]
    audio_path = voice_info["audio"]
    text_path = voice_info["text"]
    codes_path = voice_info.get("codes")

    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=500,
            detail=f"Voice audio file not found: {audio_path}"
        )

    ref_text = get_ref_text_cached(text_path) if os.path.exists(text_path) else ""

    return audio_path, ref_text, codes_path

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VieNeu-TTS API Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "voices": "/api/voices",
            "synthesize": "/api/synthesize",
            "batch_synthesize": "/api/batch-synthesize",
            "load_model": "/api/load-model"
        }
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global tts_instance, current_config

    model_loaded = tts_instance is not None
    backend = "LMDeploy" if current_config["use_lmdeploy"] else "Standard"

    return HealthResponse(
        status="healthy" if model_loaded else "not_ready",
        model_loaded=model_loaded,
        backend=backend,
        backbone=current_config["backbone"],
        codec=current_config["codec"]
    )

@app.get("/api/voices")
async def list_voices():
    """List available voices"""
    voices = []
    for name, info in VOICE_SAMPLES.items():
        voices.append({
            "name": name,
            "audio_path": info["audio"],
            "text_path": info["text"],
            "available": os.path.exists(info["audio"])
        })

    return {
        "voices": voices,
        "count": len(voices)
    }

@app.post("/api/load-model")
async def load_model(config: ModelConfigRequest):
    """Load TTS model with specified configuration"""
    global tts_instance, current_config

    try:
        # Validate configuration
        if config.backbone not in BACKBONE_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid backbone. Available: {list(BACKBONE_CONFIGS.keys())}"
            )

        if config.codec not in CODEC_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid codec. Available: {list(CODEC_CONFIGS.keys())}"
            )

        # Cleanup previous instance
        if tts_instance is not None:
            del tts_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load configuration
        backbone_config = BACKBONE_CONFIGS[config.backbone]
        codec_config = CODEC_CONFIGS[config.codec]

        # Determine if LMDeploy should be used
        use_lmdeploy = config.use_lmdeploy and should_use_lmdeploy(config.device)

        # Determine devices
        if config.device == "Auto":
            if "gguf" in config.backbone.lower():
                if sys.platform == "darwin":
                    backbone_device = "gpu"
                else:
                    backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
            else:
                if sys.platform == "darwin":
                    backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                else:
                    backbone_device = "cuda" if torch.cuda.is_available() else "cpu"

            if "ONNX" in config.codec:
                codec_device = "cpu"
            elif sys.platform == "darwin":
                codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
            else:
                codec_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            backbone_device = config.device.lower()
            codec_device = config.device.lower()
            if "ONNX" in config.codec:
                codec_device = "cpu"

        # Load model
        if use_lmdeploy:
            backbone_device = "cuda"
            codec_device = "cuda" if "ONNX" not in config.codec else "cpu"

            tts_instance = FastVieNeuTTS(
                backbone_repo=backbone_config["repo"],
                backbone_device=backbone_device,
                codec_repo=codec_config["repo"],
                codec_device=codec_device,
                memory_util=0.3,
                tp=1,
                enable_prefix_caching=True,
                enable_triton=True,
            )

            # Pre-cache voice references
            for voice_name, voice_info in VOICE_SAMPLES.items():
                audio_path = voice_info["audio"]
                text_path = voice_info["text"]
                if os.path.exists(audio_path) and os.path.exists(text_path):
                    ref_text = get_ref_text_cached(text_path)
                    tts_instance.get_cached_reference(voice_name, audio_path, ref_text)
        else:
            tts_instance = VieNeuTTS(
                backbone_repo=backbone_config["repo"],
                backbone_device=backbone_device,
                codec_repo=codec_config["repo"],
                codec_device=codec_device
            )

        # Update current config
        current_config = {
            "backbone": config.backbone,
            "codec": config.codec,
            "device": config.device,
            "use_lmdeploy": use_lmdeploy
        }

        return {
            "status": "success",
            "message": "Model loaded successfully",
            "backend": "LMDeploy" if use_lmdeploy else "Standard",
            "backbone": config.backbone,
            "codec": config.codec
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/api/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text using specified voice.

    Returns WAV audio file directly.
    """
    tts = get_tts_instance()

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Get voice information
        audio_path, ref_text, codes_path = get_voice_info(request.voice)

        # Encode reference or load preencoded codes
        codec_config = CODEC_CONFIGS[current_config["codec"]]
        use_preencoded = codec_config.get('use_preencoded', False)

        if use_preencoded and codes_path and os.path.exists(codes_path):
            ref_codes = torch.load(codes_path, map_location="cpu", weights_only=True)
        else:
            # Use cached reference if available (FastVieNeuTTS only)
            if current_config["use_lmdeploy"] and hasattr(tts, 'get_cached_reference'):
                ref_codes = tts.get_cached_reference(request.voice, audio_path, ref_text)
            else:
                ref_codes = tts.encode_reference(audio_path)

        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()

        # Split text into chunks
        text_chunks = split_text_into_chunks(request.text.strip(), max_chars=MAX_CHARS_PER_CHUNK)

        # Synthesize
        all_audio_segments = []
        sr = 24000
        silence_pad = np.zeros(int(sr * 0.15), dtype=np.float32)

        # Use batch processing if available
        if current_config["use_lmdeploy"] and hasattr(tts, 'infer_batch') and len(text_chunks) > 1:
            chunk_wavs = tts.infer_batch(text_chunks, ref_codes, ref_text)
            for i, chunk_wav in enumerate(chunk_wavs):
                if chunk_wav is not None and len(chunk_wav) > 0:
                    all_audio_segments.append(chunk_wav)
                    if i < len(text_chunks) - 1:
                        all_audio_segments.append(silence_pad)
        else:
            # Sequential processing
            for i, chunk in enumerate(text_chunks):
                chunk_wav = tts.infer(chunk, ref_codes, ref_text)
                if chunk_wav is not None and len(chunk_wav) > 0:
                    all_audio_segments.append(chunk_wav)
                    if i < len(text_chunks) - 1:
                        all_audio_segments.append(silence_pad)

        if not all_audio_segments:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # Concatenate audio
        final_wav = np.concatenate(all_audio_segments)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            tmp_path = tmp.name

        # Read file and return as response
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        # Cleanup
        os.unlink(tmp_path)

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/api/batch-synthesize", response_model=BatchSynthesizeResponse)
async def batch_synthesize(request: BatchSynthesizeRequest):
    """
    Synthesize multiple texts in batch.

    Returns JSON with base64-encoded audio for each text.
    """
    tts = get_tts_instance()

    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    try:
        # Get voice information
        audio_path, ref_text, codes_path = get_voice_info(request.voice)

        # Encode reference or load preencoded codes
        codec_config = CODEC_CONFIGS[current_config["codec"]]
        use_preencoded = codec_config.get('use_preencoded', False)

        if use_preencoded and codes_path and os.path.exists(codes_path):
            ref_codes = torch.load(codes_path, map_location="cpu", weights_only=True)
        else:
            if current_config["use_lmdeploy"] and hasattr(tts, 'get_cached_reference'):
                ref_codes = tts.get_cached_reference(request.voice, audio_path, ref_text)
            else:
                ref_codes = tts.encode_reference(audio_path)

        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()

        results = []
        total_duration = 0.0
        sr = 24000

        # Process each text
        for text in request.texts:
            text = text.strip()
            if not text:
                continue

            # Split into chunks
            text_chunks = split_text_into_chunks(text, max_chars=MAX_CHARS_PER_CHUNK)

            # Synthesize
            all_audio_segments = []
            silence_pad = np.zeros(int(sr * 0.15), dtype=np.float32)

            if current_config["use_lmdeploy"] and hasattr(tts, 'infer_batch') and len(text_chunks) > 1:
                chunk_wavs = tts.infer_batch(text_chunks, ref_codes, ref_text)
                for i, chunk_wav in enumerate(chunk_wavs):
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_audio_segments.append(chunk_wav)
                        if i < len(text_chunks) - 1:
                            all_audio_segments.append(silence_pad)
            else:
                for i, chunk in enumerate(text_chunks):
                    chunk_wav = tts.infer(chunk, ref_codes, ref_text)
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_audio_segments.append(chunk_wav)
                        if i < len(text_chunks) - 1:
                            all_audio_segments.append(silence_pad)

            if not all_audio_segments:
                continue

            final_wav = np.concatenate(all_audio_segments)
            duration = len(final_wav) / sr
            total_duration += duration

            # Encode to base64
            audio_base64 = encode_audio_to_base64(final_wav, sr)

            results.append({
                "text": text,
                "audio": audio_base64,
                "duration": duration
            })

        return BatchSynthesizeResponse(
            results=results,
            count=len(results),
            totalDuration=total_duration
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch synthesis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    # Auto-load model on startup (optional)
    auto_load = os.getenv("AUTO_LOAD_MODEL", "false").lower() == "true"
    if auto_load:
        print("Auto-loading TTS model...")
        try:
            # Default configuration
            backbone = os.getenv("DEFAULT_BACKBONE", "VieNeu-TTS-0.3B (GPU)")
            codec = os.getenv("DEFAULT_CODEC", "NeuCodec (Distill)")
            device = os.getenv("DEFAULT_DEVICE", "Auto")
            use_lmdeploy = os.getenv("USE_LMDEPLOY", "true").lower() == "true"

            # Load model synchronously at startup
            import asyncio
            config = ModelConfigRequest(
                backbone=backbone,
                codec=codec,
                device=device,
                use_lmdeploy=use_lmdeploy
            )
            asyncio.run(load_model(config))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to auto-load model: {e}")
            print("Model will need to be loaded via POST /api/load-model")

    # Run server
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)
