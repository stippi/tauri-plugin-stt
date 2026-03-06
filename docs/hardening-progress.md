# STT Plugin Hardening Progress

## Goal

Stabilize the local fork of `tauri-plugin-stt` enough for production use in the tutor app.

## Findings To Address

1. Desktop leaks immortal CPAL streams and can spawn duplicate capture pipelines after model switches.
2. Desktop stop path drops buffered recognizer state and can lose the tail of an utterance.
3. Desktop availability/permission reporting is misleading.
4. Android permission flow is incomplete and likely leaves `requestPermission()` unresolved.
5. iOS shared mutable state is accessed from multiple queues despite comments claiming serialization.
6. Tauri iOS base listener storage remains a separate framework-level risk.

## Plan

1. Fix desktop stream ownership and finalization.
2. Fix Android permission lifecycle and tighten command behavior.
3. Serialize iOS plugin state access inside the plugin where possible.
4. Re-run validation and document remaining framework limitations.

## Progress Log

- 2026-03-06: Created tracking note and started implementation.
- 2026-03-06: Desktop fixes landed. Reused the existing audio pipeline across model switches, flushed `final_result()` on stop, and made desktop availability reflect whether an input device exists.
- 2026-03-06: Attempted to override iOS listener registration/removal in the plugin, but Tauri exposes those methods as `internal` in the Swift package, so the plugin cannot legally override or call through to them from a separate package. Reverted the change to keep iOS builds green.
- 2026-03-06: Android permission flow no longer relies on an uncalled private handler; it now exposes an `onRequestPermissionsResult` hook and uses a polling fallback to resolve pending permission requests.

## Validation

- `cargo check`: passes
- `make ios` from the tutor app: used to catch Swift visibility/override issues in the plugin
- iOS Swift changes: not compiled in this session
- Android Kotlin changes: not compiled in this session

## Remaining Known External Risk

- Tauri iOS base `Plugin` listener storage is not owned by this plugin. Even after plugin-local hardening, listener register/remove behavior may still require an app-side workaround or a Tauri patch.
