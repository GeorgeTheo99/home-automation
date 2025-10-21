import io
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from openwakeword import Model as OWWModel


# ----------------- Config -----------------
load_dotenv()

HA_URL = os.getenv("HA_URL", "http://127.0.0.1:8123").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5")

WAKEWORD = os.getenv("WAKEWORD", "hey_jarvis")
WAKEWORD_MODEL = os.getenv("WAKEWORD_MODEL", "")
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.6"))
MIC_DEVICE_INDEX = None if os.getenv("MIC_DEVICE_INDEX", "default") == "default" else int(os.getenv("MIC_DEVICE_INDEX"))
POST_WAKE_RECORD_SECS = float(os.getenv("POST_WAKE_RECORD_SECS", "2.5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
SPEAK_ENABLED = os.getenv("SPEAK", "true").lower() == "true"
TTS_ENGINE = os.getenv("TTS_ENGINE", "espeak")  # espeak or espeak-ng
TTS_VOICE = os.getenv("TTS_VOICE", "en-us")
TTS_RATE = int(os.getenv("TTS_RATE", "170"))
IGNORE_AFTER_SPEAK_SECS = float(os.getenv("IGNORE_AFTER_SPEAK_SECS", "2.0"))
MIC_DEVICE_NAME = os.getenv("MIC_DEVICE_NAME", "").strip()
INPUT_GAIN = float(os.getenv("INPUT_GAIN", "1.0"))
WAKE_COOLDOWN_SEC = float(os.getenv("WAKE_COOLDOWN_SEC", "3.0"))
WAKE_STREAK = int(os.getenv("WAKE_STREAK", "2"))
WAKE_HYSTERESIS_SEC = float(os.getenv("WAKE_HYSTERESIS_SEC", "0.8"))
NLU_MODE = os.getenv("NLU_MODE", "fast_first").lower()  # fast_only | fast_first | gpt_only
LOG_NLU = os.getenv("LOG_NLU", "true").lower() == "true"

SAMPLE_RATE = 16000
CHANNELS = 1
WAKE_BLOCK_SEC = float(os.getenv("WAKE_BLOCK_SEC", "0.5"))
STREAM_BLOCK_SEC = float(os.getenv("STREAM_BLOCK_SEC", "0.2"))


def log(level: str, *args):
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        print(f"[{level}]", *args)


# ----------------- OpenAI + HTTP -----------------
if not OPENAI_API_KEY:
    log("WARN", "OPENAI_API_KEY not set; STT/NLU will fail.")
oai = OpenAI(api_key=OPENAI_API_KEY)

sess = requests.Session()
if HA_TOKEN:
    sess.headers.update({
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    })


# ----------------- Entity aliases -----------------
ALIASES_STATIC = {
    r"\b(kitchen (light|lights))\b": "light.kitchen",
    r"\bliving( |_)room (light|lights)\b": "light.living_room",
    r"\bbed(room)? (light|lights)\b": "light.bedroom",
}

ALIASES_DYNAMIC: List[Tuple[re.Pattern, str]] = []
ENT_FRIENDLY: Dict[str, str] = {}
NLU_ENTITY_CONTEXT: List[Dict[str, object]] = []


def slugify_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _name_variants(name: str) -> List[str]:
    s = slugify_name(name)
    basic = s
    nospace = s.replace(" ", "")
    under = s.replace(" ", "_")
    variants = [basic, nospace, under]
    # Provide both with and without trailing light(s)
    if not basic.endswith(" light") and not basic.endswith(" lights"):
        variants += [f"{basic} light", f"{basic} lights"]
    return sorted(set(variants), key=len, reverse=True)


def build_entity_aliases() -> None:
    global ALIASES_DYNAMIC
    global NLU_ENTITY_CONTEXT
    ALIASES_DYNAMIC = []
    NLU_ENTITY_CONTEXT = []
    try:
        url = f"{HA_URL}/api/states"
        r = sess.get(url, timeout=5)
        r.raise_for_status()
        entities = r.json()
        for e in entities:
            if not isinstance(e, dict):
                continue
            ent_id = e.get("entity_id", "")
            if not (ent_id.startswith("light.") or ent_id.startswith("switch.") or ent_id.startswith("scene.")):
                continue
            attrs = e.get("attributes", {}) or {}
            fname = attrs.get("friendly_name")
            if not fname:
                continue
            ENT_FRIENDLY[ent_id] = fname
            alias_variants = _name_variants(fname)
            for v in alias_variants:
                pat = re.compile(rf"\b{re.escape(v)}\b", re.I)
                ALIASES_DYNAMIC.append((pat, ent_id))
            NLU_ENTITY_CONTEXT.append(
                {
                    "entity_id": ent_id,
                    "friendly_name": fname,
                    "domain": ent_id.split(".", 1)[0] if "." in ent_id else None,
                    "room": attrs.get("room_name") or attrs.get("area_id"),
                    "area_name": attrs.get("area_name"),
                    "aliases": alias_variants,
                    "raw_attributes": {k: v for k, v in attrs.items() if k in {"room_name", "area_id", "area_name", "entity_picture"}},
                }
            )
        log("INFO", f"Built {len(ALIASES_DYNAMIC)} entity alias patterns from HA.")
    except Exception as ex:
        log("WARN", f"Alias build failed: {ex}")


def resolve_entity(utterance: str) -> Optional[str]:
    u = utterance.lower()
    # static first
    for pat, ent in [(re.compile(p, re.I), e) for p, e in ALIASES_STATIC.items()]:
        if re.search(pat, u):
            return ent
    # dynamic from HA
    for pat, ent in ALIASES_DYNAMIC:
        if re.search(pat, u):
            return ent
    # heuristics
    if "kitchen" in u and "light" in u:
        return "light.kitchen"
    if "living" in u and "light" in u:
        return "light.living_room"
    if "bed" in u and "light" in u:
        return "light.bedroom"
    return None


# ----------------- Wake word -----------------
def load_oww() -> OWWModel:
    try:
        if WAKEWORD_MODEL and os.path.exists(WAKEWORD_MODEL):
            log("INFO", f"Loading wakeword model: {WAKEWORD_MODEL}")
            return OWWModel(wakeword_models=[WAKEWORD_MODEL])
        log("INFO", "Loading default OpenWakeWord models")
        return OWWModel()
    except Exception as ex:
        log("ERROR", f"OpenWakeWord init failed: {ex}")
        raise


oww = load_oww()


def check_wakeword(scores: Dict[str, float]) -> bool:
    # Prefer explicit label match when present
    for label, val in scores.items():
        if label == WAKEWORD or label.startswith(f"{WAKEWORD}_"):
            if val >= WAKE_THRESHOLD:
                return True
    # Fallback: any strong "hey_*" model
    for k, v in scores.items():
        if k.startswith("hey_") and v >= WAKE_THRESHOLD:
            return True
    return False


# ----------------- Audio helpers -----------------
def record_seconds(seconds: float) -> np.ndarray:
    frames = int(seconds * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", device=MIC_DEVICE_INDEX)
    sd.wait()
    return audio.flatten()


def pcm16_to_wav_bytes(pcm16_array: np.ndarray, samplerate: int = SAMPLE_RATE) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm16_array.tobytes())
    return buf.getvalue()


# ----------------- STT (Whisper API) -----------------
def stt_whisper_api(pcm16: np.ndarray) -> str:
    wav = pcm16_to_wav_bytes(pcm16)
    bio = io.BytesIO(wav)
    bio.name = "speech.wav"
    tx = oai.audio.transcriptions.create(model="whisper-1", file=bio)
    return (getattr(tx, "text", None) or "").strip()


# ----------------- NLU via GPT -----------------
SYSTEM_PROMPT_BASE = """You are a Home Assistant command formatter.
Input: a short utterance like 'kitchen lights 50%' or 'kitchen lights blue'.
Output: ONLY a compact JSON object with keys: domain, service, entity_id, data.
Rules:
- Use correct Home Assistant domain/service (e.g., light.turn_on, light.turn_off).
- Pick entity_id values ONLY from the provided Valid entities list. Do not invent new ids.
- If the user requests all lights or a group of devices, return a list of entity_ids from the valid list or a provided group entity.
- For percentages, use {"brightness_pct": X} 0..100.
- For colors, use {"color_name": "<basic-color>"}.
- For OFF, use service 'turn_off'.
- Do not add commentary or code fences.
"""


def nlu_llm(utterance: str) -> Dict:
    """Call the chat completion API to get strict JSON. Handles models that require default temperature."""
    temperature = float(os.getenv("NLU_TEMPERATURE", "1"))
    model = os.getenv("MODEL_NAME", MODEL_NAME)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
    ]
    if NLU_ENTITY_CONTEXT:
        context_json = json.dumps(NLU_ENTITY_CONTEXT, ensure_ascii=False)
        messages.append({
            "role": "system",
            "content": f"Valid entities (JSON array): {context_json}",
        })
    messages.append({"role": "user", "content": utterance})
    kwargs = dict(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    try:
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = oai.chat.completions.create(**kwargs)
    except Exception as e:
        # Retry without temperature if model rejects non-default values
        if "unsupported_value" in str(e) or "temperature" in str(e):
            resp = oai.chat.completions.create(**{k: v for k, v in kwargs.items() if k != "temperature"})
        else:
            raise
    txt = resp.choices[0].message.content
    return json.loads(txt)


# ----------------- HA call -----------------
ALLOWED = {
    ("light", "turn_on"),
    ("light", "turn_off"),
    ("switch", "turn_on"),
    ("switch", "turn_off"),
    ("scene", "turn_on"),
}


def ha_call(domain: str, service: str, entity_id: Optional[object], data: Optional[Dict] = None) -> List[dict]:
    if (domain, service) not in ALLOWED:
        raise ValueError(f"Blocked service: {domain}.{service}")
    url = f"{HA_URL}/api/services/{domain}/{service}"
    payload: Dict[str, object] = {}

    if entity_id is None:
        pass
    elif isinstance(entity_id, (list, tuple)):
        ents_set = set()
        ents: List[str] = []
        for ent in entity_id:
            if not isinstance(ent, str):
                continue
            if not re.match(r"^[a-z_]+\.[a-z0-9_\.]+$", ent):
                raise ValueError(f"Invalid entity_id format: {ent}")
            if ent not in ents_set:
                ents_set.add(ent)
                ents.append(ent)
        if not ents:
            raise ValueError("Empty entity_id list")
        payload["entity_id"] = ents
    elif isinstance(entity_id, str):
        if not re.match(r"^[a-z_]+\.[a-z0-9_\.]+$", entity_id):
            raise ValueError("Invalid entity_id format")
        payload["entity_id"] = entity_id
    else:
        raise ValueError("Unsupported entity_id type")

    if data:
        payload.update(data)
    r = sess.post(url, json=payload, timeout=7)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return []


# ----------------- Local TTS -----------------
import shutil
import subprocess


def _friendly(entity_id: str) -> str:
    if entity_id in ENT_FRIENDLY:
        return ENT_FRIENDLY[entity_id]
    try:
        domain, name = entity_id.split(".", 1)
        return name.replace("_", " ")
    except Exception:
        return entity_id


def confirm_message(cmd: Dict) -> str:
    domain = cmd.get("domain", "")
    service = cmd.get("service", "")
    ent = cmd.get("entity_id", "")
    data = cmd.get("data", {}) or {}
    name = _friendly(ent)
    if domain == "light":
        if service == "turn_off":
            return f"Turning off {name}."
        if service == "turn_on":
            if "brightness_pct" in data:
                return f"Setting {name} to {int(data['brightness_pct'])} percent."
            if "color_name" in data:
                return f"Setting {name} to {data['color_name']}."
            return f"Turning on {name}."
    if domain == "switch":
        if service == "turn_off":
            return f"Turning off {name}."
        if service == "turn_on":
            return f"Turning on {name}."
    if domain == "scene" and service == "turn_on":
        return f"Activating scene {name}."
    return "Done."


MUTE_UNTIL = 0.0


def speak(text: str) -> None:
    if not SPEAK_ENABLED:
        return
    # Prefer espeak-ng if available, else espeak
    engine = None
    for candidate in [TTS_ENGINE, "espeak-ng", "espeak"]:
        if candidate and shutil.which(candidate):
            engine = candidate
            break
    if not engine:
        log("WARN", f"No TTS engine found; skipping speech: {text}")
        return
    try:
        cmd = [engine]
        if TTS_VOICE:
            cmd.append(f"-v{TTS_VOICE}")
        cmd.append(f"-s{TTS_RATE}")
        cmd.append(text)
        subprocess.run(cmd, check=False)
    except Exception as ex:
        log("WARN", f"TTS failed: {ex}")


# ----------------- Fast path -----------------
PAT_ON = re.compile(r"\b(on)\b", re.I)
PAT_OFF = re.compile(r"\b(off)\b", re.I)
PAT_PCT = re.compile(r"(\d{1,3})\s?%")
COLOR_WORDS = {"red", "green", "blue", "yellow", "purple", "pink", "orange", "white", "cyan", "magenta"}


def fast_path(utterance: str) -> Optional[Dict]:
    u = utterance.lower().strip()
    ent = resolve_entity(u)
    if not ent:
        return None
    if PAT_OFF.search(u):
        return {"domain": "light", "service": "turn_off", "entity_id": ent, "data": {}}
    if PAT_ON.search(u):
        return {"domain": "light", "service": "turn_on", "entity_id": ent, "data": {}}
    m = PAT_PCT.search(u)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return {"domain": "light", "service": "turn_on", "entity_id": ent, "data": {"brightness_pct": pct}}
    for c in COLOR_WORDS:
        if c in u:
            return {"domain": "light", "service": "turn_on", "entity_id": ent, "data": {"color_name": c}}
    return None


def _entities_for_domain(domain: str) -> List[str]:
    return [e["entity_id"] for e in NLU_ENTITY_CONTEXT if e.get("domain") == domain]


def normalize_command(cmd: Dict) -> Dict:
    if not cmd:
        return cmd
    domain = cmd.get("domain")
    entity = cmd.get("entity_id")

    if isinstance(entity, list):
        cleaned = []
        for ent in entity:
            if isinstance(ent, str):
                ent = ent.strip()
                if ent and "." not in ent:
                    resolved = resolve_entity(ent)
                    if resolved:
                        ent = resolved
                    else:
                        continue
                if ent and "." in ent:
                    cleaned.append(ent)
        if cleaned:
            cmd["entity_id"] = cleaned
        return cmd

    if not isinstance(entity, str):
        return cmd

    ent_str = entity.strip()
    if not ent_str:
        return cmd

    lower = ent_str.lower()
    if domain == "light" and lower in {"all", "all_lights", "all lights", "lights"}:
        group = next((e["entity_id"] for e in NLU_ENTITY_CONTEXT if e["entity_id"] == "light.all_lights"), None)
        if group:
            cmd["entity_id"] = group
        else:
            lights = _entities_for_domain("light")
            if lights:
                cmd["entity_id"] = lights
        return cmd

    if "." not in ent_str:
        # try alias resolution
        resolved = resolve_entity(ent_str)
        if resolved:
            cmd["entity_id"] = resolved
        return cmd

    return cmd


# ----------------- Main loop -----------------
def main():
    global MUTE_UNTIL
    if not HA_TOKEN:
        log("ERROR", "HA_TOKEN is not set")
        return
    build_entity_aliases()
    # Show devices for convenience and select mic if name is provided
    device_index = MIC_DEVICE_INDEX
    try:
        devs = sd.query_devices()
        names = [d["name"] for d in devs]
        log("INFO", f"Audio devices: {names}")
        if MIC_DEVICE_NAME and device_index is None:
            low = MIC_DEVICE_NAME.lower()
            for i, d in enumerate(devs):
                if low in (d["name"] or "").lower() and d.get("max_input_channels", 0) > 0:
                    device_index = i
                    log("INFO", f"Selected MIC_DEVICE_NAME match index {i}: {d['name']}")
                    break
        if device_index is None:
            log("INFO", f"Using system default input device")
        else:
            log("INFO", f"Using input device index {device_index}: {devs[device_index]['name']}")
    except Exception as e:
        log("WARN", f"Audio device enumeration failed: {e}")

    log("INFO", "Voice gateway ready. Say the wake word…")
    stream_block = max(1, int(STREAM_BLOCK_SEC * SAMPLE_RATE))
    post_frames = int(POST_WAKE_RECORD_SECS * SAMPLE_RATE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", device=device_index, blocksize=stream_block) as stream:
        streak = 0
        hysteresis_until = 0.0
        while True:
            # Read a short chunk continuously
            frames, overflowed = stream.read(stream_block)
            samples_i16 = np.asarray(frames).flatten()
            samples_f32 = samples_i16.astype(np.float32) / 32768.0
            if INPUT_GAIN != 1.0:
                samples_f32 = np.clip(samples_f32 * INPUT_GAIN, -1.0, 1.0)

            now = time.monotonic()
            if now < MUTE_UNTIL or now < hysteresis_until:
                continue

            rms = 0.0
            if LOG_LEVEL == "DEBUG":
                try:
                    rms = float(np.sqrt(np.mean(samples_f32 ** 2)))
                except Exception:
                    rms = 0.0

            scores = oww.predict(samples_f32)
            if LOG_LEVEL == "DEBUG":
                tops = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                log("DEBUG", f"RMS={rms:.4f} top={tops} thr={WAKE_THRESHOLD}")

            if check_wakeword(scores):
                streak += 1
            else:
                streak = 0

            if streak >= WAKE_STREAK:
                streak = 0
                # Immediately set cooldown so we don't re-trigger while recording/speaking
                cooldown = max(WAKE_COOLDOWN_SEC, POST_WAKE_RECORD_SECS + 0.5)
                MUTE_UNTIL = now + cooldown
                hysteresis_until = now + cooldown + WAKE_HYSTERESIS_SEC
                log("INFO", "Wake word detected.")
                # Capture post-wake audio from the same stream
                collected = []
                remaining = post_frames
                while remaining > 0:
                    n = min(stream_block, remaining)
                    f2, _ = stream.read(n)
                    collected.append(np.asarray(f2).flatten())
                    remaining -= n
                pcm16 = np.concatenate(collected).astype(np.int16)

                log("INFO", "Transcribing…")
                try:
                    text = stt_whisper_api(pcm16)
                except Exception as ex:
                    log("ERROR", f"STT error: {ex}")
                    continue
                if not text:
                    log("WARN", "Empty transcription.")
                    continue
                log("INFO", f"Heard: {text}")

                # NLU selection
                cmd = None
                nlu_source = "fast-path"
                try:
                    if NLU_MODE == "gpt_only":
                        nlu_source = "gpt"
                        cmd = nlu_llm(text)
                    elif NLU_MODE == "fast_only":
                        cmd = fast_path(text)
                    else:  # fast_first
                        cmd = fast_path(text)
                        if cmd is None:
                            nlu_source = "gpt"
                            cmd = nlu_llm(text)
                except Exception as ex:
                    log("ERROR", f"NLU error: {ex}")
                    continue

                if cmd is None:
                    log("WARN", "No command produced by NLU")
                    continue
                original_entity = cmd.get("entity_id")
                cmd = normalize_command(cmd)
                normalized_entity = cmd.get("entity_id")
                if LOG_NLU:
                    log("INFO", f"NLU: source={nlu_source} cmd={cmd}")
                    if normalized_entity != original_entity:
                        log("INFO", f"NLU normalize: entity_id {original_entity!r} -> {normalized_entity!r}")

                try:
                    log("INFO", f"HA call: {cmd}")
                    ha_call(cmd["domain"], cmd["service"], cmd["entity_id"], cmd.get("data", {}))
                    msg = confirm_message(cmd)
                    speak(msg)
                    MUTE_UNTIL = max(MUTE_UNTIL, time.monotonic() + IGNORE_AFTER_SPEAK_SECS)
                except Exception as ex:
                    log("ERROR", f"HA call failed: {ex}")


if __name__ == "main" or __name__ == "__main__":
    main()
