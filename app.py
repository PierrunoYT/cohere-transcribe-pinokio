import gradio as gr
import torch
import tempfile
import os
import time
from pathlib import Path

from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

# Model configuration
MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"

SUPPORTED_LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Spanish": "es",
    "Portuguese": "pt",
    "Greek": "el",
    "Dutch": "nl",
    "Polish": "pl",
    "Arabic": "ar",
    "Vietnamese": "vi",
    "Chinese (Mandarin)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
}

# Global model cache
_model_cache = {}


def get_model(device="auto", hf_token=None):
    """Load model with caching to avoid reloading on every request."""
    cache_key = hf_token or "no_token"
    if cache_key not in _model_cache:
        print("Loading Cohere Transcribe model...")
        auth_kwargs = {"token": hf_token} if hf_token else {}
        processor = AutoProcessor.from_pretrained(MODEL_ID, **auth_kwargs)
        model = CohereAsrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            **auth_kwargs,
        )
        _model_cache[cache_key] = {"processor": processor, "model": model}
        print("Model loaded successfully!")
    return _model_cache[cache_key]["processor"], _model_cache[cache_key]["model"]


def download_model(hf_token, progress=gr.Progress()):
    """Pre-download and cache the model so first transcription is faster."""
    progress(0, desc="Validating token...")
    token = (hf_token or "").strip() or None
    try:
        progress(0.2, desc="Downloading model files...")
        get_model(hf_token=token)
        progress(1.0, desc="Done")
        return "Model is downloaded and ready."
    except Exception as e:
        return f"Model download failed: {str(e)}"


def transcribe_audio(
    audio_file, language, punctuation, hf_token, progress=gr.Progress()
):
    """Transcribe an audio file."""
    if audio_file is None:
        return "Please upload an audio file.", ""

    progress(0, desc="Loading model...")
    token = (hf_token or "").strip() or None
    processor, model = get_model(hf_token=token)

    progress(0.3, desc="Loading audio...")
    try:
        audio = load_audio(audio_file, sampling_rate=16000)
    except Exception as e:
        return f"Error loading audio: {str(e)}", ""

    lang_code = SUPPORTED_LANGUAGES.get(language, "en")

    progress(0.5, desc="Processing audio...")
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        language=lang_code,
        punctuation=punctuation,
    )
    inputs.to(model.device, dtype=model.dtype)

    progress(0.7, desc="Generating transcription...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    elapsed = time.time() - start_time

    text = processor.decode(outputs[0], skip_special_tokens=True)

    stats = f"Transcribed in {elapsed:.2f}s"
    return text, stats


def transcribe_long_audio(
    audio_file, language, punctuation, hf_token, progress=gr.Progress()
):
    """Transcribe long-form audio with automatic chunking."""
    if audio_file is None:
        return "Please upload an audio file.", ""

    progress(0, desc="Loading model...")
    token = (hf_token or "").strip() or None
    processor, model = get_model(hf_token=token)

    progress(0.2, desc="Loading audio...")
    try:
        audio = load_audio(audio_file, sampling_rate=16000)
    except Exception as e:
        return f"Error loading audio: {str(e)}", ""

    lang_code = SUPPORTED_LANGUAGES.get(language, "en")
    duration_s = len(audio) / 16000

    progress(0.4, desc="Processing audio...")
    inputs = processor(
        audio=audio,
        sampling_rate=16000,
        return_tensors="pt",
        language=lang_code,
        punctuation=punctuation,
    )
    audio_chunk_index = inputs.get("audio_chunk_index")
    inputs.to(model.device, dtype=model.dtype)

    progress(0.6, desc="Generating transcription...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    elapsed = time.time() - start_time

    text = processor.decode(
        outputs[0],
        skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index,
        language=lang_code,
    )

    rtfx = duration_s / elapsed if elapsed > 0 else 0
    stats = f"Audio duration: {duration_s / 60:.1f} min | Transcribed in {elapsed:.1f}s | RTFx: {rtfx:.1f}x"
    return text, stats


def create_demo():
    """Create the Gradio demo interface."""
    with gr.Blocks(title="Cohere Transcribe") as demo:
        gr.Markdown(
            """
            # 🎙️ Cohere Transcribe
            State-of-the-art open-source speech recognition model supporting 14 languages.
            **2B parameter ASR model from Cohere Labs** | Apache 2.0 License
            """
        )

        with gr.Row():
            hf_token = gr.Textbox(
                label="Hugging Face Token (for gated model access)",
                type="password",
                placeholder="hf_...",
            )
            download_btn = gr.Button("Download Model", variant="secondary")
        download_status = gr.Textbox(
            label="Model Download Status", lines=2, interactive=False
        )

        download_btn.click(
            fn=download_model,
            inputs=[hf_token],
            outputs=[download_status],
        )

        with gr.Tabs():
            with gr.Tab("Short-form"):
                gr.Markdown(
                    "Transcribe audio files (recommended for clips under 30 seconds)"
                )

                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Upload Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        language = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value="English",
                            label="Language",
                        )
                        punctuation = gr.Checkbox(
                            label="Include Punctuation", value=True
                        )
                        transcribe_btn = gr.Button("Transcribe", variant="primary")

                    with gr.Column():
                        text_output = gr.Textbox(label="Transcription", lines=8)
                        stats_output = gr.Textbox(
                            label="Statistics", lines=2, interactive=False
                        )

                transcribe_btn.click(
                    fn=transcribe_audio,
                    inputs=[audio_input, language, punctuation, hf_token],
                    outputs=[text_output, stats_output],
                )

            with gr.Tab("Long-form"):
                gr.Markdown("Transcribe long audio files with automatic chunking")

                with gr.Row():
                    with gr.Column():
                        audio_input_long = gr.Audio(
                            label="Upload Audio", type="filepath", sources=["upload"]
                        )
                        language_long = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value="English",
                            label="Language",
                        )
                        punctuation_long = gr.Checkbox(
                            label="Include Punctuation", value=True
                        )
                        transcribe_btn_long = gr.Button(
                            "Transcribe Long Audio", variant="primary"
                        )

                    with gr.Column():
                        text_output_long = gr.Textbox(label="Transcription", lines=12)
                        stats_output_long = gr.Textbox(
                            label="Statistics", lines=2, interactive=False
                        )

                transcribe_btn_long.click(
                    fn=transcribe_long_audio,
                    inputs=[
                        audio_input_long,
                        language_long,
                        punctuation_long,
                        hf_token,
                    ],
                    outputs=[text_output_long, stats_output_long],
                )

        gr.Markdown(
            """
            ### Supported Languages
            English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish,
            Arabic, Vietnamese, Chinese (Mandarin), Japanese, Korean
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
