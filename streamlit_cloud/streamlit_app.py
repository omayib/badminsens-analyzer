"""Streamlit Community Cloud entrypoint: runs ../app.py.

The app lives in this folder only so Community Cloud picks up streamlit_cloud/requirements.txt
(no torch: SVM stroke model) instead of the root requirements.txt (CUDA torch for training).
Deploy settings: main file streamlit_cloud/streamlit_app.py (Python 3.12, as tested).
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
