"""Entry point for the CLI interface."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import validate_config
from interfaces.cli import run

validate_config()

if __name__ == "__main__":
    run()
