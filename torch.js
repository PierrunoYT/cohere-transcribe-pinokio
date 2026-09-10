// Keep hardware selection in one place; all builds resolve their dependencies.
module.exports = {
  run: [
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "python -c \"raise SystemExit('Intel macOS is unsupported: Transformers 5 requires a newer PyTorch than its available wheels.')\""
      },
      next: null
    },
    {
      when: "{{gpu === 'nvidia' && (platform === 'win32' || platform === 'linux')}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128"
      },
      next: null
    },
    {
      when: "{{gpu === 'amd' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/rocm6.3"
      },
      next: null
    },
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.7.0"
      },
      next: null
    },
    {
      // Includes AMD Windows: the app does not implement DirectML inference.
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu"
      }
    }
  ]
}
