# BladeVision Ground-Truth Mod (Phase 0, lab-only)

Read-only Fabric client mod that exports **per-tick ground-truth game state** — self and nearest
opponent position/rotation/velocity/health/hit-registration — to a JSONL file, wall-clock stamped
so it aligns with the Python recorder's frames.

It is *lab instrumentation*, not part of any bot: it never emits input or affects the game, and it
stays completely inert unless the `BLADEVISION_LAB` environment variable is set. Its job is to
**auto-label the recorded pixels** with real geometry (making perception training cheap) and to
give detectors a ground-truth reference (docs/DESIGN.md §6).

## Toolchain

Matched to the user's other MC 26.x mods: Minecraft 26.1.2, Fabric Loader 0.19.3, Loom 1.17,
Fabric API 0.154.2, Java 25, Mojang mappings.

## Build

```bash
cd mod
./gradlew build
# remapped jar -> build/libs/bladevision-truth-0.1.0.jar
```

Copy the jar into your MC instance's `mods/` folder (alongside Fabric API).

## Run (enable logging)

Launch the client with the lab gate set:

```bash
export BLADEVISION_LAB=1
export BLADEVISION_TRUTH_DIR="/absolute/path/to/BladeVision/data/truth"   # optional; default: <gamedir>/bladevision
# then start Minecraft from that same shell / launcher environment
```

Each session writes `truth_<timestamp>.jsonl`, one JSON object per client tick:

```json
{"t_wall":1732200000123,"t_nano":...,"tick":12345,
 "self":{"x":..,"y":..,"z":..,"vx":..,"vy":..,"vz":..,"yaw":..,"pitch":..,
         "health":20.0,"onGround":true,"sprinting":false,"hurtTime":0,"item":"item.minecraft.diamond_sword","food":20},
 "opponent":{...,"dist":3.2}}
```

## How it pairs with the recorder

Run the Python recorder (`python -m recorder.record`) and this mod at the same time. The recorder
captures frames + discrete inputs; the mod captures rotation + positions + health + hits. Their
`t_wall` timestamps are resampled onto a common grid offline to build the training dataset
(docs/DESIGN.md §2.2). **Rotation/aim ground-truth comes from this mod**, because when Minecraft
grabs the cursor the OS-level input hook can't see in-game look movement.

## Notes / TODO

- Sub-frame clock sync via an on-screen flash marker (§2.2) is a later refinement; wall-clock
  alignment on a single machine is accurate to a few ms and is enough for the first dataset.
- Opponent on-screen bounding box is derived offline from world position + camera geometry; it is
  intentionally not computed in the mod to keep it minimal.
