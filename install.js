module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "notify",
      params: {
        html: "Installing Cohere Transcribe..."
      }
    },
    // Install PyTorch with GPU support first, so the correct build is
    // present before any other dependency can pull in a generic default
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
    // Install remaining dependencies
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
    {
      method: "notify",
      params: {
        html: "✅ Installed! Model downloads on first launch (~4GB). ~8GB VRAM recommended."
      }
    }
  ]
}
