# Voice Gateway Progress — 2025-10-22

## Summary
- Restored reliable wake-word scoring and standardized threshold logging.
- Improved realtime capture robustness and softened handling of “commit empty”.
- Enabled true barge‑in (cuts off all assistant playback immediately).
- Added a distinct negative chime when no speech is detected after wake.
- Tuned follow‑up sensitivity and realtime VAD parameters.

## Fixes and Enhancements

- Wake‑word loop and logs
  - Restored the per‑chunk wake loop so `openwakeword.predict(...)` and `Scores=...` run continuously during capture.
  - Standardized score/threshold logging with consistent precision and included both `thr` and `final` thresholds.

- Realtime robustness
  - Client‑managed audio capture for follow‑ups (disabled server VAD by default) to avoid timing edge cases.
  - Soft‑handle `input_audio_buffer_commit_empty`: if any transcript/voice activity already arrived, don’t abort the turn; finalize gracefully instead.
  - Prevent “noise‑only” cancellation when a transcript exists.
  - Diagnostics: persist captured audio WAVs for unhandled realtime sessions under `diagnostics/realtime_captures/`.

- Follow‑ups
  - Removed the extra post‑playback extension of `ASSISTANT_SPEAK_UNTIL` so listening opens immediately after assistant audio completes.
  - Lowered follow‑up RMS/peak gates to make follow‑ups easier under normal speaking volume.

- True barge‑in (all playback)
  - Local TTS runs asynchronously and is interruptible.
  - New `stop_tts()` stops local TTS and calls `sd.stop()` to immediately cut realtime/chime playback; clears `ASSISTANT_SPEAK_UNTIL` and `MUTE_UNTIL` to resume listening.
  - Capture loop monitors for user speech while assistant audio is active and triggers barge‑in on peak/RMS thresholds.

- Negative chime for “no speech”
  - Added a short, descending “negative” chime played when no speech is detected after wake or when sessions are dropped for noise.
  - Supports custom WAV via env; otherwise a synthesized beep is generated.

## Configuration Changes (.env)

- Wake thresholds
  - `WAKE_THRESHOLD=0.0010`
  - `WAKE_FINAL_THRESHOLD=0.0012`

- Realtime/VAD
  - `REALTIME_SERVER_VAD=false`
  - `REALTIME_PREFETCH_SEC=0.7`
  - `REALTIME_MIN_INPUT_AUDIO_MS=2000`
  - `REALTIME_VAD_SILENCE_MS=700`
  - `REALTIME_VAD_THRESHOLD=0.38`
  - `REALTIME_NOISE_PEAK=0` (disabled during calibration)

- Follow‑ups & barge‑in
  - `FOLLOWUP_MIN_RMS=0.010`
  - `FOLLOWUP_MIN_PEAK=2000`
  - `BARGE_IN_ENABLED=true`

- Negative chime (defaults)
  - `NEGATIVE_CHIME_ENABLED=true`
  - `NEGATIVE_CHIME_FREQ=440.0`, `NEGATIVE_CHIME_DURATION=0.18`, `NEGATIVE_CHIME_VOLUME=0.25`

## Behavior Notes

- Saying the wake phrase and then remaining silent produces a quick negative chime rather than silence.
- Speaking over assistant audio immediately stops playback and opens the mic (true barge‑in) for the next utterance.
- Realtime sessions no longer fail a turn if a transcript already arrived but a late `commit_empty` error is emitted.

## Next Suggestions (optional)

- Enable PipeWire WebRTC echo cancellation and point the mic input to the echo‑canceled source to further reduce false wakes/barge‑in from our own audio.
- Two‑stage wake verify (e.g., a tiny verifier or quick ASR check for the target phrase) if false wakes persist with stricter thresholds.

