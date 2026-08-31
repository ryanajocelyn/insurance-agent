"""
Pytest configuration file to configure sys.path for source modules.
"""

import sys
from pathlib import Path
import markupsafe
import jinja2.utils
import jinja2.ext

# Patch jinja2.utils.Markup for jinjasql compatibility with jinja2 3.1+
if not hasattr(jinja2.utils, "Markup"):
    jinja2.utils.Markup = markupsafe.Markup

# Patch jinja2.ext.autoescape for jinjasql extension loading
if not hasattr(jinja2.ext, "autoescape"):
    class AutoEscapeExtension(jinja2.ext.Extension):
        def __init__(self, environment):
            super().__init__(environment)
    jinja2.ext.autoescape = AutoEscapeExtension

# Add project root and src directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
