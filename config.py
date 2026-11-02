"""Paths. Set VZ_ROOT to the working directory (data/, models/, work/ are created beneath it)."""
import os
ROOT = os.environ.get('VZ_ROOT', os.path.dirname(os.path.abspath(__file__)))
