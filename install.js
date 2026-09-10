module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      when: "{{exists('app/env/.installed')}}",
      method: "fs.rm",
      params: {
        path: "app/env/.installed"
      }
    },
    {
      method: "notify",
      params: {
        html: "Installing Cohere Transcribe..."
      }
    },
    // Install the platform's torch build before resolving app dependencies.
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
    // Install dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip check",
          "python -c \"from app import create_demo; create_demo()\""
        ]
      }
    },
    {
      method: "fs.write",
      params: {
        path: "app/env/.installed",
        text: "ready"
      }
    },
    {
      method: "notify",
      params: {
        html: "✅ Installed! Model downloads on first use (~4GB). ~8GB VRAM recommended."
      }
    }
  ]
}
