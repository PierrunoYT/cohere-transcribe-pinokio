# Cohere Transcribe

A Gradio-based web interface for [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), a state-of-the-art open-source speech recognition model supporting 14 languages.

## Features

- **14 Languages**: English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish, Arabic, Vietnamese, Chinese (Mandarin), Japanese, Korean
- **Short-form Transcription**: Quick transcription for audio clips under 30 seconds
- **Long-form Transcription**: Automatic chunking and reassembly for longer audio files
- **Punctuation Control**: Toggle punctuation on/off
- **Model Caching**: Model loads once and stays in memory for fast subsequent transcriptions

## Quick Start with Pinokio

This application is packaged for [Pinokio](https://pinokio.com/) for one-click installation and management.

**Available Commands:**
- **Install** - Sets up Python environment, installs dependencies, and configures PyTorch for your GPU
- **Start** - Launches the Gradio UI at `http://localhost:7860`
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
