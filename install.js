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
        message: "git lfs install"
      }
    },
    // Install uv if not available
    {
      method: "shell.run",
      params: {
        message: "pip install uv"
      }
    },
    // Install dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
    // Install PyTorch with GPU support
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "."
        }
      }
    },
    // Create link for UI
    {
      method: "script.start",
      params: {
        uri: "link.js"
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
