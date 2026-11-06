"""Launch the UI bound to localhost with telemetry off, whatever the current directory or user config."""
import os
import sys
from pathlib import Path

from streamlit.web import cli

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    os.chdir(ROOT)
    sys.argv = ["streamlit", "run", str(ROOT / "promopilot_ui" / "app.py"),
                "--server.address", "localhost", "--browser.serverAddress", "localhost",
                "--browser.gatherUsageStats", "false", *sys.argv[1:]]
    sys.exit(cli.main())
