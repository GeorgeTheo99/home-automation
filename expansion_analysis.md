# Voice Assistant Expansion Analysis

## Current Voice Gateway Capabilities
- Voice pipeline: wake word → speech capture → OpenAI ASR → GPT intent formatter → Home Assistant REST (`voice-gateway/main.py`).
- Supported domains: `light`, `switch`, `scene.turn_on`; scene activation already routed via aliases/LLM.
- Fast-path heuristics map literal commands (on/off, numeric brightness, basic colors) when an entity alias resolves.
- Real-time prompt enforces terse English confirmations and discourages stylistic replies.
- Confirmation strings are static and deterministic; no persona layer or contextual flair.

## Scene Authoring & Advanced Control
- Extend `ALLOWED` service list and add a dedicated tool schema for `scene.create`/`scene.apply`.
- Decide on persistence: call HA’s `scene.create` (ephemeral) or update project YAML + trigger `scene.reload`; latter may require a helper service with file locking/validation.
- Enrich entity context with capability metadata (supports color, temperature, effects) to guide GPT when crafting scene payloads.
- Script multi-turn flows (“What should we call this scene?”, “Which rooms?”) and play back a summary before saving.
- Validate inputs: ensure at least one light is present, clamp brightness/colors, reject unsupported attributes, and surface errors conversationally.

## Personality & Creative Lighting
- Introduce a persona specification (traits, tone, catchphrases) fed into both realtime and fallback prompts.
- Replace rigid confirmations with persona-aware responses: either let GPT draft the acknowledgement post-tool call or build a template library keyed by intent/context.
- Map abstract adjectives (“cool”, “cozy”, “energize”) to curated lighting recipes; expose these in context so GPT can reuse or remix them.
- Allow richer HA payloads (`rgb_color`, `color_temp_kelvin`, `effect`) after capability checks; expand fast-path rules accordingly.
- Maintain short-term memory per session (e.g., last palette) to support follow-ups like “try a different cool look.”

## Weather, Timers/Alarms, and Web Access
- Weather: surface a `get_weather` tool backed by HA `weather.` entities or a new microservice returning structured forecast data for GPT summarization.
- Timers/alarms: integrate HA timers or a local scheduler service; expose `create_timer`, `cancel_timer`, `list_timers`, and broadcast state changes for reminders.
- Web access: add a guarded retrieval tool (allow-listed URLs or a local proxy) with explicit prompt rules for citing sources, handling failures, and respecting rate limits.
- Implement per-tool rate limiting, logging, and fallback messaging to avoid silent failures.

## Implementation Considerations
- Prompt updates: merge persona guidelines, tool usage instructions, and fallback behaviors into both realtime and legacy system prompts.
- Tool schemas: keep parameters minimal, validate payloads server-side, and document failure codes for the assistant to translate into user-friendly replies.
- Testing: simulate voice flows for creative lighting, scene creation dialogs, weather queries, and timers; add regression coverage for fast-path heuristics.
- Telemetry & observability: log tool invocations, scene creations, and persona-driven responses to monitor success rates and user satisfaction signals.

## Suggested Next Steps
1. Choose a scene persistence strategy and sketch validation rules.
2. Draft the persona config and update prompts in a feature branch; review conversational tone via dry runs.
3. Design JSON schemas and backend adapters for weather, timers, and web retrieval before expanding the allow list.
4. Prototype creative lighting recipes and ensure capability metadata is available from Home Assistant entities.
