"""
Multi-Agent Motor Claim Adjudication Assistant - Source Package.
"""

import sys
from pathlib import Path

# Add src and project root to sys.path to allow flexible package imports
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
