# Home Automation Server

Local-first Python FastAPI server to control Wi‑Fi lights and home speakers, with a future voice assistant powered by GPT‑5. Privacy by default, modular drivers for different devices, and a simple HTTP/WebSocket API for apps, automations, and voice control.

Note on current plan
- We are running Home Assistant only while we get lighting rock‑solid. The custom FastAPI agent (voice + Spotify/Tidal) will be developed separately and added later.

## Highlights
- Python 3.11+ with FastAPI + asyncio
- Auto‑discovery of new devices on the LAN (mDNS/SSDP)
- Control lights on your Wi‑Fi network (groups, scenes, schedules)
- Control home speakers (play/pause, volume, multi‑room where supported)
- Optional MQTT event bus for external automations
- Web and CLI control, plus REST + WebSocket API
- Future: on‑device wake‑word, local STT/TTS, GPT‑5 assistant for natural commands

## Goals and Non‑Goals
- Goals
  - Reliable local control without cloud dependencies where possible
  - Extensible adapter model for lights and speakers
  - Human‑readable configuration with safe defaults
  - Observable and debuggable (structured logs, metrics endpoints)
- Non‑Goals (initially)
  - Full smart‑home platform parity; this is focused on lights + speakers first
  - WAN exposure by default; remote access requires explicit opt‑in via reverse proxy/VPN

## Planned Device Support
- Lights (priority targets)
  - LIFX (LAN protocol via aiolifx)
  - TP‑Link Kasa (local control via python‑kasa)
  - SmartLife/Tuya (local control via tinytuya; falls back to cloud where necessary)
  - MQTT (e.g., Tasmota, ESPHome)
  - Philips Hue (via bridge REST API) — optional
- Speakers / Music
  - Chromecast / Google Cast (pychromecast)
  - DLNA/UPnP Media Renderers (async‑upnp‑client)
  - Sonos (SoCo/UPnP)
  - Spotify
    - Playback via local Spotify Connect endpoint (librespot container/binary)
    - Remote control via Spotify Web API (user account authorization)
  - TIDAL
    - Library/control via tidalapi (user account authorization)
    - Playback via casting (Chromecast) or DLNA bridge (e.g., upmpdcli) where applicable

Notes
- TIDAL Connect SDK is proprietary; initial TIDAL support focuses on catalog/control and playback via supported casting/bridging targets.
- Spotify Connect playback will be provided by a bundled librespot instance the server can control.

Each integration lands behind a well‑defined adapter interface. If your device isn’t listed, open an issue with details.

## Architecture
- Core Server (FastAPI + asyncio)
  - HTTP + WebSocket API (control, state, events)
  - Event bus (internal pub/sub; optional MQTT bridge)
  - Discovery (mDNS/Bonjour, SSDP) with device registry and cache
  - Persistence (YAML/JSON for config, SQLite for state/history)
- Adapters (Drivers)
  - Lights: lifx, kasa, tuya, mqtt, hue
  - Speakers: chromecast, dlna, sonos, spotify‑connect (via librespot)
  - Music Services: spotify (Web API), tidal (tidalapi)
- Automations
  - Schedules, scenes, triggers, conditional logic
- Voice Assistant (Future)
  - Wake word: openwakeword/Porcupine
  - STT: Whisper (local or remote)
  - TTS: Piper/Coqui
  - LLM: GPT‑5 integration for intent + dialogue

## API Sketch (Initial)
- Discovery / Devices
  - GET /api/devices → list all known devices (type, id, capabilities)
  - POST /api/discovery/scan → trigger active discovery
- Lights
  - GET /api/lights → list devices and state
  - POST /api/lights/:id/state { on, brightness, color } → set state
  - POST /api/scenes/apply { sceneId }
- Speakers
  - GET /api/speakers → list devices and state
  - POST /api/speakers/:id/play { uri? }
  - POST /api/speakers/:id/pause
  - POST /api/speakers/:id/volume { level }
- Outlets
  - GET /api/outlets → list outlets and state
  - POST /api/outlets/:id/state { on }
- Music Services
  - GET /api/music/search?service=spotify|tidal&q=...
  - POST /api/music/play { service, uri|trackId, target }  # target = speaker id or connect endpoint
  - POST /api/music/auth/spotify|tidal { code|credentials }  # OAuth or app‑token flow
- Events
  - WS /api/events → device updates, discovery changes, automation events

Example:
```bash
# Toggle a light on
curl -X POST http://localhost:8123/api/lights/living-room/state \
  -H 'Content-Type: application/json' \
  -d '{"on":true,"brightness":80}'

# Set speaker volume
curl -X POST http://localhost:8123/api/speakers/kitchen/volume \
  -H 'Content-Type: application/json' \
  -d '{"level":35}'

# Play a Spotify URI on local Connect endpoint
curl -X POST http://localhost:8123/api/music/play \
  -H 'Content-Type: application/json' \
  -d '{"service":"spotify","uri":"spotify:track:1xQ6trAsedVPCdbtDAmk0c","target":"living-room"}'

# Toggle an outlet
curl -X POST http://localhost:8123/api/outlets/kasa-<id>/state \
  -H 'Content-Type: application/json' \
  -d '{"on":true}'

# Control a LIFX bulb (by MAC-based id)
curl -X POST http://localhost:8123/api/lights/lifx-<mac>/state \
  -H 'Content-Type: application/json' \
  -d '{"on":true, "brightness":65, "color":"#FFA000"}'
```

## Configuration
- Files
  - config/devices.yaml → static devices, names, rooms, groups
  - config/scenes.yaml → scene definitions
  - config/automations.yaml → schedules and triggers
- Environment variables
  - HOME_AUTOMATION_PORT (default: 8123)
  - HOME_AUTOMATION_BIND (default: 0.0.0.0)
  - HOME_AUTOMATION_DISCOVERY (default: true)
  - HOME_AUTOMATION_MQTT_URL (optional, e.g., mqtt://user:pass@host:1883)
  - HOME_AUTOMATION_LOG_LEVEL (default: info)
  - SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET (for Web API auth)
  - SPOTIFY_REDIRECT_URI (for OAuth, e.g., http://localhost:8123/callback/spotify)
  - TIDAL_USERNAME / TIDAL_PASSWORD (for tidalapi; or OAuth when available)
  - TIDAL_TOKEN (optional; app token required by tidalapi)
  - LIBRESPOT_ENABLED=true|false (start local Connect endpoint)
  - LIBRESPOT_DEVICE_NAME (displayed name for Connect)

Example `config/scenes.yaml` snippet:
```yaml
- id: cozy_evening
  name: Cozy Evening
  lights:
    living-room:
      on: true
      brightness: 30
      color: warm
    hallway:
      on: false
```

## Directory Layout (proposed)
- server/ → FastAPI app, WebSocket, core runtime
- adapters/
  - lights/ (lifx, kasa, tuya, mqtt, hue)
  - speakers/ (chromecast, dlna, sonos, spotify_connect)
  - music/ (spotify_web, tidal)
- discovery/ → mDNS/SSDP drivers, device registry/cache
- automations/ → schedules, triggers, scenes engine
- state/ → persistence, models
- web/ → optional web UI
- cli/ → command‑line tools
- config/ → user configuration files
- docs/ → integration notes, API docs
- scripts/ → dev/ops utilities

This repo currently contains the README and plan. Code will be added incrementally following this structure.

## Quickstart (TBD / Plan)
Until the implementation lands, this section outlines the path to first run. Stack: Python (FastAPI + asyncio) with modular adapters.

1) Prerequisites
- Python 3.11+
- Make/Taskfile or simple scripts
- Optional: Docker (Compose) for MQTT broker, testing

2) Development Run (planned)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HOME_AUTOMATION_PORT=8123
python -m server  # starts API + discovery
```

3) Docker (Home Assistant only)
```yaml
services:
  homeassistant:
    image: ghcr.io/home-assistant/home-assistant:stable
    container_name: homeassistant
    network_mode: host
    environment:
      - TZ=${TZ:-UTC}
    volumes:
      - ./ha-config:/config
    restart: unless-stopped

  # Optional services you can enable later via profiles
  mqtt:
    image: eclipse-mosquitto:2
    container_name: mqtt
    profiles: ["mqtt"]
    ports:
      - "1883:1883"
    restart: unless-stopped
```

## Security Notes
- Keep the server on a trusted LAN; do not expose directly to the internet
- If remote access is needed, prefer a VPN or a reverse proxy with TLS, auth, and IP filtering
- Use per‑adapter credentials with least privilege (e.g., Hue bridge users)
- Be aware of multicast/broadcast protocols (mDNS, SSDP) and IGMP snooping on your switches

## Troubleshooting
- Discovery
  - Ensure your network allows mDNS (UDP 5353) and SSDP (UDP 1900)
  - Avoid double NAT; consider `network_mode: host` with Docker on Linux
- MQTT
  - Confirm broker reachability and auth; test with `mosquitto_sub`/`mosquitto_pub`
- Chromecast/DLNA
  - Same subnet routing and multicast visibility are critical for discovery
- LIFX
  - Ensure bulbs are on the same subnet; broadcast/UDP must be allowed
- Kasa (TP‑Link)
  - Newer firmware requires local control; use python‑kasa discovery
- Tuya/SmartLife
  - Local keys may be required; tinytuya can derive keys with cloud credentials

- LIFX
  - No credentials required on LAN. Ensure bulbs are reachable (same subnet) and that multicast/broadcast is not blocked.
  - Colors accept named values (red, green, blue, white, warm, cool) or hex (#RRGGBB).

### Current Deployment
- Only Home Assistant is deployed for now (lights, scenes, automations). The custom FastAPI agent will be added later for voice and music services.
- Bluetooth integration is disabled to avoid noisy logs on hosts without a stable BT stack. You can re‑enable it later from HA’s UI (Settings → Devices & Services → Integrations → Bluetooth).
- In HA, add your light integrations (Kasa, LIFX, Yeelight, Tuya, WiZ, etc.). Create a Helper → Light group (e.g., “All Lights”) to control everything at once.

## Deployment (Ubuntu)
- Recommended: Docker on Ubuntu with `network_mode: host` for proper mDNS/SSDP/casting. A starter `Dockerfile` and `docker-compose.yml` are included.
- If Docker is not installed, run `sudo ./install_docker_ubuntu.sh` (generated by setup) or install via the official Docker apt repository and ensure user `georgetheo` is in the `docker` group.
- Audio integration: enable the `librespot` service via Compose profile `audio` and map PulseAudio or ALSA devices as needed (see comments in `docker-compose.yml`).
- Discovery: Kasa (python-kasa), LIFX (lifxlan), Tuya (tinytuya). Host networking is required so UDP broadcast/multicast reaches devices.

## Roadmap
- M1: FastAPI core, discovery, device registry, adapter interface
- M2: Lights — LIFX (aiolifx), Kasa (python‑kasa), Tuya (tinytuya), scenes
- M3: Speakers — Chromecast (pychromecast), DLNA (async‑upnp‑client), Sonos (SoCo)
- M4: Music — Spotify Connect (librespot integration), Spotify Web API control
- M5: Music — TIDAL (tidalapi + casting/DLNA bridge), playlists/library
- M6: Automations (schedules, triggers), Web UI MVP
- M7: Voice pipeline (wake word, local STT/TTS), command routing
- M8: GPT‑5 integration for intent + dialogue, natural scenes/actions

## Contributing
- Propose adapters via a short design issue with device specifics
- Keep adapters stateless, idempotent, and non‑blocking (async)
- Add integration docs in `docs/` with setup steps and caveats

## Status
Active planning. This README captures the scope, architecture, and the near‑term milestones. Code scaffolding will be added next.

Deployment note: Host is Ubuntu. A system profile has been recorded at `/home/georgetheo/README.md` (used for discovery/audio planning).
