import os
from pathlib import Path


def load_local_env(start_path: str | None = None) -> None:
    """
    Lightweight .env loader.
    Loads KEY=VALUE pairs from .env if present and preserves already-exported env vars.
    """
    base = Path(start_path).resolve().parent if start_path else Path.cwd()
    env_path = base / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
