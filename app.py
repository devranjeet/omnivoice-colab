# %cd /content/omnivoice-colab
import os
import sys
import logging
from typing import Any, Dict, Optional

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile
import re
import uuid

# Audio output directory
temp_audio_dir = "./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup path for OmniVoice if running as a subfolder
# ---------------------------------------------------------------------------
OmniVoice_path = os.path.join(os.getcwd(), "OmniVoice")
if os.path.exists(OmniVoice_path) and OmniVoice_path not in sys.path:
    sys.path.append(OmniVoice_path)

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
from hf_mirror import download_model

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Model Loading (Global Scope)
# ---------------------------------------------------------------------------
print("Loading model from k2-fsa/OmniVoice to cuda ...")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

try:
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )
except Exception as e:
    logging.warning(f"Standard download failed ({e}), using hf_mirror fallback...")
    omnivoice_model_path = download_model(
        "k2-fsa/OmniVoice",
        download_folder="./OmniVoice_Model",
        redownload=False,
        workers=6,
        use_snapshot=False,
    )

    model = OmniVoice.from_pretrained(
        omnivoice_model_path,
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )

sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Event Tags & JS Functions
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", 
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", 
    "[dissatisfaction-hnn]"
]

def make_insert_tag_js(elem_id: str) -> str:
    return f"""
    (tag_val, current_text) => {{
        const textarea = document.querySelector('{elem_id} textarea');
        if (!textarea) return (current_text ? current_text + " " + tag_val : tag_val);
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        let prefix = " ";
        let suffix = " ";
        if (!current_text) return tag_val;
        if (start === 0 || current_text[start - 1] === ' ') prefix = "";
        if (end < current_text.length && current_text[end] === ' ') suffix = "";
        return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
    }}
    """

INSERT_TAG_JS_VC = make_insert_tag_js("#vc_textbox")
INSERT_TAG_JS_VD = make_insert_tag_js("#vd_textbox")

# ---------------------------------------------------------------------------
# UI Configurations & Language Mappings
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto", "English", "Hindi"]

_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
    "English Accent": [
        "Indian Accent", "American Accent", "British Accent", "Australian Accent", "Canadian Accent"
    ],
}

_ATTR_INFO = {
    "English Accent": "Only effective for English speech.",
}

# ---------------------------------------------------------------------------
# Core Logic & Helpers
# ---------------------------------------------------------------------------
def tts_file_name(text: str, language: str = "en") -> str:
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    clean_text = clean_text.lower().strip().replace(" ", "_")
    if not clean_text:
        clean_text = "audio"

    truncated = clean_text[:20]
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"
    rand = uuid.uuid4().hex[:8].upper()
    return os.path.join(temp_audio_dir, f"{truncated}_{lang}_{rand}.wav")


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
    """Core Text-to-Speech Generation Logic using OmniVoice"""
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

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
            return None, "Please upload or provide a reference audio."
        clean_ref_text = ref_text.strip() if (ref_text and ref_text.strip()) else None
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=clean_ref_text
        )
    elif mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "Done."

# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.1em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}

/* CSS for Event Tags */
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

def _lang_dropdown(label="Language (optional)", value="Auto"):
    return gr.Dropdown(
        label=label, choices=_ALL_LANGUAGES, value=value,
        allow_custom_value=False, interactive=True,
    )

def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed", info="1.0 = normal. >1 faster, <1 slower.")
        du = gr.Number(value=None, label="Duration (seconds)", info="Set a fixed duration to override speed.")
        ns = gr.Slider(4, 64, value=32, step=1, label="Inference Steps", info="Lower = faster, higher = better quality.")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
        pp = gr.Checkbox(label="Preprocess Prompt", value=True, info="Applies silence removal and trims reference audio.")
        po = gr.Checkbox(label="Postprocess Output", value=True, info="Removes long silences from generated audio.")
    return ns, gs, dn, sp, du, pp, po

with gr.Blocks(theme=theme, css=css, title="OmniVoice Hindi & English TTS") as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ OmniVoice Hindi & English TTS</h1>
            <p>High-Quality Speech Synthesis, Voice Cloning & Voice Design for Hindi and English.</p>
        </div>
    """)

    with gr.Tabs():
        # ==============================================================
        # Voice Clone Tab
        # ==============================================================
        with gr.TabItem("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text you want the cloned voice to say...",
                        elem_id="vc_textbox"
                    )
                    
                    # Tag Buttons for Voice Clone
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vc_text],
                                outputs=vc_text,
                                js=INSERT_TAG_JS_VC
                            )

                    vc_lang = _lang_dropdown("Language (optional)")
                    
                    vc_ref_audio = gr.Audio(
                        label="Reference Audio (3–10 seconds audio sample)",
                        type="filepath",
                        elem_classes="compact-audio"
                    )
                    
                    vc_ref_text = gr.Textbox(
                        label="Reference Text (Optional but recommended)",
                        lines=2, 
                        placeholder="Type what is spoken in the reference audio sample (improves clone quality)..."
                    )
                                        
                    vc_btn = gr.Button("Generate Cloned Voice", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="Output Audio", type="numpy")
                    vc_status = gr.Textbox(label="Status", lines=1)
                    vc_out_wav = gr.File(label="Download Generated Audio (WAV)")

            def _clone_fn(text, lang, ref_aud, ref_text, ns, gs, dn, sp, du, pp, po):
                res = _gen_core(
                    text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po,
                    mode="clone", ref_text=ref_text
                )
                if res[0] is None:
                    return None, res[1], None
                
                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                
                return audio_tuple, status, tmp_wav

            vc_btn.click(
                _clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                outputs=[vc_audio, vc_status, vc_out_wav],
            )

        # ==============================================================
        # Voice Design Tab
        # ==============================================================
        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text to synthesize...",
                        elem_id="vd_textbox"
                    )
                    
                    # Tag Buttons for Voice Design
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vd_text],
                                outputs=vd_text,
                                js=INSERT_TAG_JS_VD
                            )

                    vd_lang = _lang_dropdown(value='Auto')
                    vd_btn = gr.Button("Generate Designed Voice", variant="primary")
                    
                    with gr.Accordion("Character Voice Design", open=False):
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            default_val = "Auto"
                            if _cat == "Gender":
                                default_val = "Female"
                            elif _cat == "Age":
                                default_val = "Young Adult"
                                
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=["Auto"] + _choices,
                                    value=default_val,
                                    info=_ATTR_INFO.get(_cat)
                                )
                            )
                        
                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="Output Audio", type="numpy")
                    vd_status = gr.Textbox(label="Status", lines=1)
                    vd_out_wav = gr.File(label="Download Generated Audio (WAV)")

            def _build_instruct(groups):
                selected = [g for g in groups if g and g != "Auto"]
                if not selected:
                    return None
                return ", ".join(selected)

            def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                instruct = _build_instruct(groups)
                res = _gen_core(
                    text, lang, None, instruct, ns, gs, dn, sp, du, pp, po,
                    mode="design"
                )
                if res[0] is None:
                    return None, res[1], None
                
                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                
                return audio_tuple, status, tmp_wav

            vd_btn.click(
                _design_fn,
                inputs=[vd_text, vd_lang, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                outputs=[vd_audio, vd_status, vd_out_wav],
            )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)

