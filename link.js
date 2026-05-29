module.exports = {
  run: [
    {
      method: "fs.link",
      params: {
        path: "app",
        venv: "env"
      }
    }
  ]
}
