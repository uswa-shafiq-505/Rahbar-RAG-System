import sys
from pathlib import Path

# Appended, not inserted first: scripts/trace.py must not shadow the standard library `trace`.
sys.path.append(str(Path(__file__).resolve().parents[1]))
