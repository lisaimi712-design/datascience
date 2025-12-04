# Project: OASST1 sampling

This repo contains a small script, `data.py`, that flattens OpenAssistant/OASST1 conversations into human→assistant prompt/response pairs and writes a CSV sample.

Quick start

1. Create (optional) and activate a virtual environment.
2. Install dependencies:

```powershell
C:/Users/lisai/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m pip install -r requirements.txt
```

3. Run the entrypoint:

```powershell
C:/Users/lisai/AppData/Local/Microsoft/WindowsApps/python3.13.exe main.py
```

Files

- `data.py`: dataset loading and sampling logic.
- `main.py`: small CLI entrypoint that runs `data.py`.
- `requirements.txt`: packages needed to run the scripts.
