"""Entry point for the Streamlit interface.

Usage: streamlit run run_streamlit.py
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import validate_config

validate_config()

sys.exit(subprocess.call(["streamlit", "run", "interfaces/streamlit_app.py"]))
