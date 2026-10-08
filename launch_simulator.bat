@echo off
REM Opens the interactive simulator notebook in JupyterLab (in your browser).
cd /d "%~dp0"
python -m jupyter lab notebooks\quantum_face_simulator.ipynb
