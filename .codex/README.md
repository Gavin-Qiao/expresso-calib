# Codex Environment

Codex local environments are configured through the Codex app settings pane.
For this project, use this setup script:

```powershell
powershell -ExecutionPolicy Bypass -File tools/codex-launch-server.ps1 -Sync
```

Add a project action named `Run Server` with this command:

```powershell
powershell -ExecutionPolicy Bypass -File tools/codex-launch-server.ps1
```

The server listens on:

```text
http://127.0.0.1:3987/operator
http://127.0.0.1:3987/target
```

The setup script syncs dependencies for new worktrees. The action omits `-Sync`
so starting the server stays quick during normal robot work.
