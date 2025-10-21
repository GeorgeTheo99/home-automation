# Voice Gateway (Path A)

Wake word (local) → STT (OpenAI Whisper API) → NLU (GPT‑5 JSON) → Home Assistant REST.

## Prerequisites
- Ubuntu/Linux host with microphone access
- Home Assistant running (Container is fine) and a Long‑Lived Access Token
- Python 3.10+
- System packages
  - `sudo apt update && sudo apt install -y python3-venv python3-pip portaudio19-dev ffmpeg espeak-ng alsa-utils`

## Setup (uv recommended)
Install uv (once):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# then restart your shell or: source ~/.local/bin/env
```

Install a Python toolchain that works with OpenWakeWord (Python 3.11 recommended):
```bash
uv python install 3.11
```

Create environment and install deps:
```bash
cd /home/georgetheo/home_automation/voice-gateway
uv venv -p 3.11  # creates .venv with Python 3.11
uv sync  # installs dependencies from pyproject.toml

cp .env.example .env
# Edit .env: set HA_URL, HA_TOKEN, OPENAI_API_KEY
```

Run (no activation needed):
```bash
uv run python main.py
```

## Audio Setup (PipeWire recommended)
We configured PipeWire (user session) and set Built‑in Analog Stereo as the default sink/source.

1) Install/enable PipeWire + WirePlumber
```bash
sudo apt install -y pipewire pipewire-audio-client-libraries wireplumber
systemctl --user enable --now pipewire pipewire-pulse wireplumber
```

2) Set defaults and unmute via `wpctl`
```bash
wpctl status
# Identify your Built-in Analog Stereo sink and source IDs (e.g., sink=48, source=49)
wpctl set-default 48
wpctl set-default 49
wpctl set-mute 49 0
wpctl set-volume 49 0.85
```

3) Verify mic levels (RMS should rise when you speak)
```bash
uv run python - <<'PY'
import numpy as np, sounddevice as sd
def rms(idx):
    a = sd.rec(int(0.5*16000), samplerate=16000, channels=1, dtype='int16', device=idx); sd.wait()
    x = a.flatten().astype('float32')/32768.0
    return float(np.sqrt((x*x).mean()))
print('Default devices (in,out):', sd.default.device)
for i in [13,5]:
    try: print(i, 'RMS=', rms(i))
    except Exception as e: print(i, 'error:', e)
PY
```

If you prefer ALSA‑only (no PipeWire), use `alsamixer -c 0` to enable/unmute Capture and `arecord -D default` to test.

## Alternative (pip/venv)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: list microphones with indices and defaults
```bash
uv run python - <<'PY'
import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    print(i, d['name'], 'in_channels=', d.get('max_input_channels', 0))
print('Default devices (in,out):', sd.default.device)
PY
# Then set MIC_DEVICE_INDEX or MIC_DEVICE_NAME in .env
```

## Run
Using uv:
```bash
uv run python main.py
```
or with an activated venv:
```bash
source .venv/bin/activate
python main.py
```
Speak the wake word (e.g., "hey jarvis"), then try:
- "kitchen lights on"
- "kitchen lights 50 percent"
- "living room lights off"

If your entity IDs differ, the gateway builds aliases from HA friendly names automatically at startup. You can also add/override regexes in `ALIASES_STATIC` inside `main.py`.

## Systemd (optional)
Create a user service:
```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/ha-voice-gateway.service <<'UNIT'
[Unit]
Description=HA Voice Gateway (wake word -> STT -> GPT-5 -> HA)
After=network-online.target sound.target

[Service]
Type=simple
WorkingDirectory=%h/home_automation/voice-gateway
Environment=PYTHONUNBUFFERED=1
ExecStart=%h/home_automation/voice-gateway/.venv/bin/python %h/home_automation/voice-gateway/main.py
Restart=on-failure

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable ha-voice-gateway
systemctl --user start ha-voice-gateway
journalctl --user -u ha-voice-gateway -f
```

## Notes
- This gateway avoids HA’s Assist/Wyoming UI entirely.
- STT uses OpenAI Whisper API; no local model cache required.
- NLU uses `MODEL_NAME` (default: `gpt-5`). If unavailable, set to `gpt-4o-mini` or `gpt-4.1`.
  - Some models enforce default temperature; set `NLU_TEMPERATURE=1.0` (default) or let the app auto-retry without temperature.
- The gateway allow‑lists HA services for safety (lights/switches by default).
- Local speak-back uses `espeak-ng` by default. You can adjust voice/rate via `.env`.
- GPT NLU gets a JSON array of valid entities (friendly names, room/area, aliases) extracted from HA at startup. Keep entity names in HA up to date for best results.

### NLU Modes
- `fast_only`: Only use local regex fast-paths (on/off, brightness %, simple colors) with HA entity aliasing from friendly names.
- `fast_first`: Try fast-paths, fall back to GPT JSON if no match. (default)
- `gpt_only`: Always call GPT for NLU; disables fast-paths.
  - Enable logs with `LOG_NLU=true` to see which path was used and the final JSON.

## Wake Word Troubleshooting
- Default wakeword: `hey_jarvis`. Saying “weather” will not trigger; only the configured wakeword (or any `hey_*` model in fallback) does.
- To narrow detection, set `WAKEWORD_MODEL` to the bundled model:
  - `voice-gateway/.venv/lib/python3.11/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.tflite`
- Tune sensitivity: try `WAKE_THRESHOLD=0.10..0.55`, `WAKE_BLOCK_SEC=1.0`, `STREAM_BLOCK_SEC=0.2`.
- With `LOG_LEVEL=DEBUG`, you’ll see `RMS` and top scores. RMS should rise when you speak.
- If RMS is low and scores stay tiny, increase `INPUT_GAIN` (e.g., 2.0–6.0). Keep RMS generally below ~0.3 to avoid heavy clipping.
- Avoid double triggers: increase `WAKE_COOLDOWN_SEC` (e.g., >= `POST_WAKE_RECORD_SECS + 0.5`) and/or require a short streak with `WAKE_STREAK=2`.
- Add hysteresis: `WAKE_HYSTERESIS_SEC` adds extra guard time after cooldown to ignore trailing echoes/background voice.

## Remove old Whisper cache (manual)
If present, remove the legacy cache directory used for offline STT:
```bash
rm -rf /home/georgetheo/home_automation/whisper-cache
```
