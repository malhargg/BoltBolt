from __future__ import annotations

import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from streamlit.web import cli as streamlit_cli

URL = "http://127.0.0.1:8501"


def bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def runtime_root() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def server_is_running() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8501), timeout=0.35):
            return True
    except OSError:
        return False


def open_browser_when_ready() -> None:
    for _ in range(60):
        if server_is_running():
            webbrowser.open(URL)
            return
        time.sleep(0.5)


def main() -> None:
    root = bundle_root()
    app_path = root / "app.py"
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "false")

    if server_is_running():
        webbrowser.open(URL)
        return

    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_path = runtime_root() / "launcher_error.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
