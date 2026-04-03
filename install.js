module.exports = {
  run: [
    {
      method: "notify",
      params: {
        html: "Installing Cohere Transcribe..."
      }
    },
    // Install Git LFS for large model files
    {
      method: "shell.run",
      params: {
        path: ".",
        message: "git lfs install"
      }
    },
    // Install dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Install PyTorch with GPU support
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "notify",
      params: {
        html: "✅ Installed! Model downloads on first launch (~4GB). ~8GB VRAM recommended."
      }
    }
  ]
}
