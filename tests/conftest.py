import os
import sys

# Ensure repository root (project) is on sys.path so 'src' package can be imported during pytest collection
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
