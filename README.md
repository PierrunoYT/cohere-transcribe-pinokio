# Cohere Transcribe

A Gradio-based web interface for [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), a state-of-the-art open-source 2B parameter conformer-based encoder-decoder speech recognition model supporting 14 languages.

## Features

- **14 Languages**: English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish, Arabic, Vietnamese, Chinese (Mandarin), Japanese, Korean
- **Short-form Transcription**: Quick transcription for audio clips under 30 seconds
- **Long-form Transcription**: Automatic chunking and reassembly for longer audio files
- **Punctuation Control**: Toggle punctuation on/off
- **Model Caching**: Model loads once and stays in memory for fast subsequent transcriptions

## Project layout

Pinokio launcher scripts live in the repository root; application code lives under `app/`:

```
project-root/
├── app/
│   ├── app.py              # Gradio entry point
│   └── requirements.txt    # Python dependencies (PyTorch via torch.js)
├── install.js / start.js   # Pinokio launcher scripts
├── pinokio.js / pinokio.json
└── README.md
```

## Quick Start with Pinokio

This application is packaged for [Pinokio](https://pinokio.com/) for one-click installation and management.

**Available Commands:**
- **Install** - Sets up Python environment, installs dependencies, and configures PyTorch for your GPU
- **Start** - Launches the Gradio UI on `127.0.0.1` using the next free port; Pinokio shows **Open Web UI** with the exact URL
- **Update** - Pulls latest changes from repository
- **Reset** - Removes environment for clean reinstall

## System Requirements

**Minimum:**
- Python 3.10 or higher (3.12 recommended)
- 16GB RAM
- 20GB free disk space
- Internet connection for model download (~4GB on first run)

**Recommended:**
- 32GB RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.x compatible drivers

**Note:** CPU-only mode is supported but significantly slower.

## Programmatic access

### Pinokio (JavaScript)

Launcher scripts are Node modules that export `module.exports` with Pinokio steps (`shell.run`, `local.set`, `script.start`, and others). See `install.js`, `start.js`, `update.js`, `reset.js`, and `pinokio.js` in this repository. For the full script API, see the **Programming Pinokio** section in Pinokio’s `PINOKIO.md` (bundled with the Pinokio app).

### Python (Transformers)

Use the same model ID as the Gradio app. For preprocessing, decoding, and long-form chunking, mirror the logic in `app/app.py`.

```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
import torch

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = CohereAsrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)
# Load 16 kHz mono audio, build inputs with processor(...), then model.generate(...).
```

### Python (Gradio Client)

When the server is running, you can call the live app with [`gradio_client`](https://www.gradio.app/docs/python-client/client):

```python
from gradio_client import Client

client = Client("http://127.0.0.1:<PORT>")  # use the URL from Pinokio after Start
# client.view_api()  # list endpoints for your Gradio version
```

### curl

Gradio serves a local HTTP API. After **Start**, use the base URL from Pinokio (not a fixed port). Discover routes for your Gradio version, for example:

```bash
curl -s "http://127.0.0.1:<PORT>/gradio_api/info"
```

Replace `<PORT>` with the port from your **Open Web UI** link. If the path differs, check the Gradio version’s API docs or your browser’s network tab while using the UI.

## Usage Guide

### Short-form Transcription

1. Select the "Short-form" tab
2. Upload an audio file or record via microphone
3. Select the audio language
4. Toggle punctuation if needed
5. Click "Transcribe"

### Long-form Transcription

1. Select the "Long-form" tab
2. Upload a longer audio file (e.g., meetings, podcasts, earnings calls)
3. Select the audio language
4. Click "Transcribe Long Audio"
5. The model automatically chunks and reassembles the transcription

## Supported Languages

| Language | Code |
|----------|------|
| English | en |
| French | fr |
| German | de |
| Italian | it |
| Spanish | es |
| Portuguese | pt |
| Greek | el |
| Dutch | nl |
| Polish | pl |
| Arabic | ar |
| Vietnamese | vi |
| Chinese (Mandarin) | zh |
| Japanese | ja |
| Korean | ko |

## Memory Usage

| Mode | VRAM Required |
|------|---------------|
| GPU (float16) | ~8GB |
| CPU (float32) | ~16GB RAM |

## Troubleshooting

**Out of Memory Errors**
- Close other GPU applications
- Use CPU mode if GPU memory is insufficient

**Installation Issues**
- Run **Reset** in Pinokio
- Run **Install** again
- Check Python version compatibility

**Model Download Failures**
- Verify internet connection
- Ensure sufficient disk space (~20GB)
- Check firewall settings

**Poor Transcription Quality**
- Ensure audio is in the selected language
- Use clear audio with minimal background noise
- Try enabling/disabling punctuation

## Best Practices

1. **Select correct language** - The model does not auto-detect language
2. **Quality audio** - Use clear recordings with minimal background noise
3. **Monolingual audio** - Model performs best on single-language audio
4. **VAD preprocessing** - For noisy audio, consider using a voice activity detector to remove silence

## Resources

- **Model**: [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- **Technical Blog**: [Hugging Face Blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)
- **Announcement**: [Cohere Blog](https://cohere.com/blog/transcribe)
- **Leaderboard**: [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

## License

Apache 2.0 License (same as Cohere Transcribe model)

## Acknowledgments

- **Cohere Labs** - Model development and open-source release
- **Hugging Face** - Transformers integration and hosting
- **Gradio** - Web interface framework
