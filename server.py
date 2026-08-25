import os
import sys
import time
import uuid
import re
import shutil
import logging
import argparse
import subprocess
import threading
from typing import Optional, Dict, Any

import torch
import numpy as np
import scipy.io.wavfile as wavfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Setup Paths & Directories
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "Omni_Audio")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Add subfolder OmniVoice to sys.path if present
omni_sub = os.path.join(BASE_DIR, "OmniVoice")
if os.path.exists(omni_sub) and omni_sub not in sys.path:
    sys.path.append(omni_sub)

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from hf_mirror import download_model

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("omnivoice-server")
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
print("⏳ Loading OmniVoice model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_cached_dir = os.path.join(BASE_DIR, "OmniVoice_Model")

if os.path.exists(model_cached_dir) and len(os.listdir(model_cached_dir)) > 2:
    print(f"⚡ Loading cached OmniVoice model from: {model_cached_dir}...")
    model = OmniVoice.from_pretrained(
        model_cached_dir,
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )
else:
    try:
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype,
            load_asr=False,
        )
    except Exception as e:
        logger.warning(f"Standard load failed ({e}), using hf_mirror download...")
        model_path = download_model(
            "k2-fsa/OmniVoice",
            download_folder=model_cached_dir,
            redownload=False,
            workers=6,
            use_snapshot=False,
        )
        model = OmniVoice.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            load_asr=False,
        )

sampling_rate = model.sampling_rate
print(f"✅ OmniVoice Model Loaded Successfully on {device.upper()}! (Sample Rate: {sampling_rate}Hz)")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def tts_file_name(text: str, language: str = "en") -> str:
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    clean_text = clean_text.lower().strip().replace(" ", "_")
    if not clean_text:
        clean_text = "audio"
    truncated = clean_text[:20]
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"
    rand = uuid.uuid4().hex[:8].upper()
    return f"{truncated}_{lang}_{rand}.wav"


def _gen_core(
    text: str,
    language: Optional[str],
    ref_audio: Optional[str],
    instruct: Optional[str],
    num_step: int,
    guidance_scale: float,
    denoise: bool,
    speed: Optional[float],
    duration: Optional[float],
    preprocess_prompt: bool,
    postprocess_output: bool,
    mode: str,
    ref_text: Optional[str] = None
):
    """Core Text-to-Speech Generation Logic"""
    if not text or not text.strip():
        raise ValueError("Please enter text to synthesize.")

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            raise ValueError("Reference audio sample is required for cloning.")
        clean_ref_text = ref_text.strip() if (ref_text and ref_text.strip()) else None
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=clean_ref_text
        )
    elif mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    audio = model.generate(**kw)
    waveform = (audio[0] * 32767).astype(np.int16)
    return sampling_rate, waveform

# ---------------------------------------------------------------------------
# FastAPI Application & Routes
# ---------------------------------------------------------------------------
app = FastAPI(title="OmniVoice Studio", version="2.0")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class VoiceDesignRequest(BaseModel):
    text: str
    language: Optional[str] = "Auto"
    gender: Optional[str] = "Female"
    age: Optional[str] = "Young Adult"
    pitch: Optional[str] = "Auto"
    style: Optional[str] = "Auto"
    accent: Optional[str] = "Auto"
    speed: Optional[float] = 1.0
    num_step: Optional[int] = 32
    guidance_scale: Optional[float] = 2.0
    duration: Optional[float] = None
    denoise: Optional[bool] = True
    preprocess_prompt: Optional[bool] = True
    postprocess_output: Optional[bool] = True


@app.get("/")
async def serve_ui():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h2>OmniVoice Studio: templates/index.html not found.</h2>")


@app.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.post("/api/generate/voice-design")
async def api_voice_design(req: VoiceDesignRequest):
    try:
        attrs = [req.gender, req.age, req.pitch, req.style, req.accent]
        selected = [a for a in attrs if a and a != "Auto"]
        instruct = ", ".join(selected).lower() if selected else None

        sr, waveform = _gen_core(
            text=req.text,
            language=req.language,
            ref_audio=None,
            instruct=instruct,
            num_step=req.num_step or 32,
            guidance_scale=req.guidance_scale or 2.0,
            denoise=req.denoise,
            speed=req.speed,
            duration=req.duration,
            preprocess_prompt=req.preprocess_prompt,
            postprocess_output=req.postprocess_output,
            mode="design"
        )

        filename = tts_file_name(req.text, language=req.language or "en")
        output_filepath = os.path.join(AUDIO_DIR, filename)
        wavfile.write(output_filepath, sr, waveform)

        duration_sec = round(len(waveform) / sr, 2)
        return {
            "status": "success",
            "audio_url": f"/api/audio/{filename}",
            "filename": filename,
            "duration": duration_sec
        }

    except Exception as e:
        logger.error(f"Voice Design Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/generate/voice-clone")
async def api_voice_clone(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    language: Optional[str] = Form("Auto"),
    speed: Optional[float] = Form(1.0),
    num_step: Optional[int] = Form(32),
    guidance_scale: Optional[float] = Form(2.0),
    duration: Optional[float] = Form(None),
    denoise: Optional[bool] = Form(True),
    preprocess_prompt: Optional[bool] = Form(True),
    postprocess_output: Optional[bool] = Form(True),
):
    temp_ref_path = None
    try:
        ext = os.path.splitext(ref_audio.filename)[1] if ref_audio.filename else ".wav"
        temp_ref_path = os.path.join(AUDIO_DIR, f"temp_ref_{uuid.uuid4().hex[:6]}{ext}")
        with open(temp_ref_path, "wb") as f:
            shutil.copyfileobj(ref_audio.file, f)

        sr, waveform = _gen_core(
            text=text,
            language=language,
            ref_audio=temp_ref_path,
            instruct=None,
            num_step=num_step or 32,
            guidance_scale=guidance_scale or 2.0,
            denoise=denoise,
            speed=speed,
            duration=duration,
            preprocess_prompt=preprocess_prompt,
            postprocess_output=postprocess_output,
            mode="clone",
            ref_text=ref_text
        )

        filename = tts_file_name(text, language=language or "en")
        output_filepath = os.path.join(AUDIO_DIR, filename)
        wavfile.write(output_filepath, sr, waveform)

        duration_sec = round(len(waveform) / sr, 2)
        return {
            "status": "success",
            "audio_url": f"/api/audio/{filename}",
            "filename": filename,
            "duration": duration_sec
        }

    except Exception as e:
        logger.error(f"Voice Clone Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if temp_ref_path and os.path.exists(temp_ref_path):
            try:
                os.remove(temp_ref_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Cloudflare Tunnel Launcher
# ---------------------------------------------------------------------------
def launch_cloudflare_tunnel(port: int = 8000):
    """Starts cloudflared tunnel and prints the public trycloudflare.com URL."""
    system_is_linux = sys.platform.startswith("linux")
    # Store cloudflared in /tmp/ so we can execute it (Google Drive blocks chmod +x)
    cloudflared_bin = "/tmp/cloudflared" if system_is_linux else "cloudflared.exe"

    if system_is_linux and not os.path.exists(cloudflared_bin):
        print("⬇️ Downloading cloudflared binary for Linux/Colab to /tmp/...")
        dl_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        try:
            import urllib.request
            urllib.request.urlretrieve(dl_url, cloudflared_bin)
            os.chmod(cloudflared_bin, 0o755)
            print("✅ Cloudflared installed successfully.")
        except Exception as e:
            print(f"⚠️ Failed to download cloudflared: {e}")
            return

    if not os.path.exists(cloudflared_bin) and shutil.which("cloudflared") is None:
        print("⚠️ Cloudflared binary not found. Skipping tunnel.")
        return

    cmd = [cloudflared_bin if os.path.exists(cloudflared_bin) else "cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"]

    def run_tunnel():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            for line in proc.stderr:
                    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        url = match.group(0)
                        print("\n" + "="*65)
                        print("🌟 OMNIVOICE STUDIO PUBLIC URL (Cloudflare Tunnel) 🌟")
                        print(f"👉 {url}")
                        print("="*65 + "\n")
        except Exception as e:
            print(f"Tunnel exception: {e}")

    t = threading.Thread(target=run_tunnel, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniVoice Studio Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--tunnel", type=str, default="none", choices=["none", "cloudflare"], help="Tunnel provider")
    args = parser.parse_args()

    if args.tunnel == "cloudflare":
        launch_cloudflare_tunnel(port=args.port)

    print(f"🚀 Starting OmniVoice Studio on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
