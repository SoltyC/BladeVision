# BladeVision — Design Document

**A vision-based sword-PvP AI for Minecraft Java Edition, built as anticheat red-team / blue-team research.**

- **Version:** 0.1 (design)
- **Date:** 2026-09-21
- **Platform:** Minecraft Java Edition (desktop)
- **Owner context:** Anticheat operator studying vision-based, no-memory-read cheats and how to detect them.

---

## 0. Purpose & scope

Traditional anticheat detects combat cheating by watching what the *client process* does:
memory reads/writes, injected packets, impossible rotations (aim snapping), reach beyond
limits, and packet timing anomalies. An AI that only ever **reads the screen** and emits
**synthetic mouse/keyboard input** touches none of those surfaces. It is, from the game
process's point of view, indistinguishable from a human at a keyboard — until you look at
the *statistics* of how it plays.

This project builds that class of cheat **in a controlled lab** in order to answer the
question that matters to an anticheat operator:

> When a cheat reads no memory and injects no packets, what detectable signal is left, and
> how much of it survives to the server side where the anticheat can actually observe it?

The working bot is the means; the **detector + residual-signature catalog** is the deliverable.

### Explicit research guardrails (design constraints, not just policy)

These shape the architecture, so they belong in the design:

1. **Isolated environment only.** All training and evaluation happen on a private LAN/server
   you control — bot-vs-bot and you-vs-bot. No deployment against unwitting players on public
   servers. The recorder and executor default to a configured allow-list of server addresses.
2. **The ground-truth mod is lab instrumentation, never shipped.** It is a separate artifact
   from the executor and is compiled/loaded only for data collection and detector training.
3. **Blue-team parity.** Every attack capability added to the bot must be matched by analysis
   of the signature it leaves. "We can do X" is incomplete without "and here is how X shows up."

---

## 1. System overview

BladeVision is **not a Fabric mod**. The no-memory-read premise requires an external
process. The system is four programs plus a dataset:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Minecraft Java client (unmodified, for the "attack" configuration)   │
│    - renders frames to screen / offscreen framebuffer                 │
│    - receives OS-level HID input                                      │
└───────────────▲───────────────────────────────────┬──────────────────┘
                │ frames (screen capture)            │ synthetic input
                │                                     │
       ┌────────┴────────┐                  ┌─────────▼─────────┐
       │  PERCEPTION      │  state vector    │  EXECUTOR         │
       │  (vision model)  ├─────────────────►│  (HID emitter +   │
       └────────▲─────────┘                  │  human motor model)│
                │                            └─────────▲─────────┘
                │                                      │ actions
        ┌───────┴───────────────────────────────────┐ │
        │  POLICY (decision model)                   ├─┘
        └────────────────────────────────────────────┘

   ── lab-only path ────────────────────────────────────────────────
       ┌─────────────────────────────────────────────────────────┐
       │  Fabric GROUND-TRUTH MOD (research build only)            │
       │    exports real entity positions, rotations, hit reg,    │
       │    health/effects → JSONL log, time-synced to frames     │
       └───────────────────────┬─────────────────────────────────┘
                               │
                     ┌─────────▼──────────┐
                     │  DATASET            │
                     │ (frame, input,      │
                     │  ground-truth)      │
                     └─────────┬──────────┘
                               │
                     ┌─────────▼──────────┐    ┌───────────────────┐
                     │  TRAINING           │    │  DETECTOR /       │
                     │ (perception+policy) │    │  BLUE-TEAM HARNESS│
                     └────────────────────┘    └───────────────────┘
```

### Components

| # | Component | Language / stack | Role |
|---|-----------|------------------|------|
| 1 | **Recorder** | Python (mss/dxcam capture) + native input hook | Capture synchronized frames + raw input for training |
| 2 | **Ground-truth mod** | Java / Fabric | Lab-only export of real game state for labeling & detector training |
| 3 | **Perception model** | PyTorch → ONNX | Frames → structured game-state vector |
| 4 | **Policy model** | PyTorch → ONNX | State → action distribution |
| 5 | **Executor** | Python/Rust + OS HID | Actions → human-like mouse/keyboard events |
| 6 | **Detector harness** | Python | Match telemetry → bot-likelihood score + signature report |

---

## 2. Data collection

### 2.1 What we capture, per frame

At a fixed cadence (target 30 Hz, decoupled from render FPS):

- **Frame**: downscaled RGB crop of the play area (e.g. 320×180 or 256×256), plus a
  higher-res crop around the crosshair for fine aim signal. Store as compressed video +
  frame index, not PNG-per-frame (storage).
- **Input state** (the label for imitation learning):
  - mouse delta `(dx, dy)` since last tick (raw counts, pre-sensitivity)
  - button state: attack (LMB), use (RMB)
  - movement keys: W/A/S/D, space (jump), shift (sneak), ctrl (sprint)
  - hotbar slot, item-use edges
- **Ground-truth** (lab only, from the mod), for supervision & evaluation:
  - opponent entity: world position, velocity, screen-projected position, on-screen bbox
  - your rotation (yaw/pitch), position, velocity, on-ground
  - health, hunger, active effects, held item, shield state
  - hit registration events (you-hit-them / them-hit-you) with server tick

### 2.2 Time synchronization

The single hardest data-quality problem. Approach:

- The ground-truth mod stamps each export with the client render frame's monotonic timestamp
  **and** the current game tick, written to a JSONL sidecar.
- The recorder stamps each captured frame and each input event with the same monotonic clock
  (both processes on one machine; use `time.perf_counter_ns` / `System.nanoTime` with a
  one-time offset calibration via a shared flash marker: mod flashes a known pixel pattern,
  recorder detects it).
- Post-hoc alignment tool resamples everything onto a common 30 Hz grid; drops frames where
  drift exceeds one tick.

### 2.3 Dataset schema

```
dataset/
  session_<id>/
    video.mkv               # H.264/265, frame-indexed
    input.parquet           # per-frame input labels
    truth.jsonl             # ground-truth, mod-exported (lab only)
    meta.json               # sensitivity, GUI scale, FOV, resolution, map, kit
```

`meta.json` matters: mouse sensitivity, FOV, and GUI scale change the pixel→rotation mapping
and HUD layout. The model must either be trained across a spread of these or conditioned on them.

### 2.4 Corpus targets

- v1 imitation baseline: ~50–100 hours of clean 1v1 sword/crystal-free matches on a fixed kit.
- "thousands of matches": realistically 20–40k rounds; plan for automated round segmentation
  (detect round start/end from HUD + ground-truth) so labeling scales.

---

## 3. Perception model

**Chosen approach: modular** (detector + state head) for v1, because it is debuggable and,
critically, its intermediate outputs are exactly what the blue-team detector needs. End-to-end
(VPT-style) is a later experiment, not the baseline.

### 3.1 Inputs

- Frame stack: last *k* frames (k=4) at 30 Hz to give velocity/animation cues (wind-up,
  knockback, blocking pose).
- Optional dual-resolution: full-scene low-res + crosshair high-res crop.

### 3.2 Outputs (the game-state vector)

- Opponent present (bool) + on-screen bbox / center offset from crosshair (Δyaw, Δpitch proxy)
- Estimated distance bucket (from bbox scale + ground-truth-calibrated regression)
- Opponent pose flags: attacking/wind-up, blocking (shield/sword), sprinting, airborne
- Self HUD read: health, hunger, held item, shield up, hurt-flash (red vignette)
- Hit-marker detection (crosshair hit indicator), damage-taken vignette

### 3.3 Model & training

- Backbone: small CNN (e.g. a trimmed YOLO/EfficientNet-lite) for the detector head; a
  compact temporal head (3D conv or GRU over frame features) for pose/motion flags.
- Supervision: ground-truth mod provides screen-projected opponent boxes and state flags for
  free — no manual labeling. This is the payoff of building the mod first.
- Metric: on-screen localization error (px), Δrotation-to-target error (deg), state-flag F1.

---

## 4. Policy (decision) model

### 4.1 Inputs / outputs

- **Input**: perception state vector + short history (recurrent state or stacked).
- **Output** (per 30 Hz step):
  - Aim intent: desired `(Δyaw, Δpitch)` toward target — *intent only*; the executor turns
    intent into human-like motion (§5). Keeping aim intent separate from motor execution is
    the key design decision that makes the human-likeness controllable and analyzable.
  - Movement: distribution over W/A/S/D/jump/sprint/sneak (strafe patterns, W-tap, jump-reset).
  - Attack: fire probability (for crit timing — jump then hit on the way down, combo spacing).
  - Use: shield/block, gapple, pearl (later phases).

### 4.2 Training

1. **Behavioral cloning (v1):** supervised on human input labels. Gets a bot that fights
   passably and, importantly, inherits human timing distributions (good for evasion realism —
   and a clean baseline for the detector to try to beat).
2. **DAgger** to fix compounding error (BC drifts into states no human demoed).
3. **Self-play RL fine-tune:** reward = damage dealt − taken, round win. See §4.3.
   Risk: RL discovers superhuman-but-detectable behavior (frame-perfect timing). That is
   itself a blue-team finding — measure it (§7.3).

### 4.3 Reinforcement learning strategy

Pure RL from pixels in Minecraft is famously sample-hungry (millions–billions of frames),
which is why OpenAI's VPT did **behavioral cloning first, then RL fine-tune**. We follow the
same recipe, with three throughput decisions that make RL tractable *without* CUDA:

- **BC prior, then RL.** BC gives human-like priors (good for evasion realism); RL explores
  the capability ceiling. Never RL from scratch on pixels.
- **Decouple perception from policy.** Do **not** run the vision model inside the RL loop.
  Train perception separately (supervised, from mod labels), then run RL on the *policy* over
  the ground-truth **state vector** — skipping the image forward-pass and render variance. At
  deploy, swap ground-truth state → perception output; close the clean-vs-noisy gap by
  injecting perception-like noise into the state during RL (domain randomization).
- **Privileged / asymmetric setup.** The critic and reward use ground-truth mod state; the
  actor uses only what will be available at deploy. The ground-truth mod graduates from
  read-only instrumentation to **environment controller** *for the training config only* —
  it supplies reward (damage dealt − taken, hit reg, win) and reset control (teleport,
  restore kit/health, round-end detection). The deployed bot remains vision-only.

**Algorithm:** PPO (stable for self-play). **Self-play against a pool of past checkpoints**
(AlphaStar-league / PSRO style) to avoid overfitting a single opponent and rock-paper-scissors
strategy cycles.

**Fast sim before the real client.** Even privileged state-vector RL against the live client
is bottlenecked at ~real-time. So the first RL work happens in a **lightweight standalone
sword-PvP simulator** (`sim/`) that models the relevant 1.9+ combat mechanics (reach, attack
cooldown/charge, sprint knockback + sprint-reset, crits, knockback) and runs far faster than
real-time on CPU. Policies trained there transfer to the client-driven env later, or seed it.
This is **Milestone A** (§10).

---

## 5. Executor & human motor model

This is where evasion is won or lost, and where the most interesting detection signal lives.

The executor converts **aim intent** into OS-level `HID` mouse motion and click events. A
naive "set rotation to target" is instantly detectable even without memory reads — the *shape*
of the motion is wrong. The motor model must reproduce human aim kinematics:

- **Reaction latency**: sample from a human-like distribution (log-normal, ~150–300 ms mean),
  not a constant. Constant latency is a tell.
- **Flick kinematics**: minimum-jerk / two-phase (ballistic + corrective) trajectories with
  overshoot and settle, not linear or instant snaps.
- **Micro-jitter & drift**: hand tremor, sub-pixel noise, small idle drift.
- **Click timing**: CPS with human variance and refractory structure — not a fixed interval
  (fixed interval = autoclicker signature).
- **Imperfection budget**: occasional misses, late reactions, target-switch hesitation.

Executor backends (choose per OS; macOS dev, likely Windows target for realism):
- Windows: `SendInput` raw mouse, or a virtual HID device for lower-level realism.
- macOS (dev/testing): `CGEvent`.

> **Design note:** the fidelity of this motor model *is* the evasion capability. It is also
> the exact thing the detector attacks. Build the two against each other.

---

## 6. Ground-truth mod (Fabric, lab only)

Minimal, research-only Fabric mod. Responsibilities:

- On each client tick, export the ground-truth record (§2.1) to JSONL.
- Emit the sync flash marker for clock calibration (§2.2).
- **Never** emits input or influences the game. It is read-only instrumentation.
- Gated behind an env var / config so it cannot be accidentally shipped with the executor.

This is what makes supervised perception cheap (auto-labels) and gives the detector a
ground-truth reference to measure "how superhuman is this behavior, really."

---

## 7. Blue-team: detector & signature catalog

The deliverable. Given a match's **observable telemetry**, output a bot-likelihood score and
a per-signal breakdown. Two observation regimes — this distinction is the core research output:

- **Client-side observable** (what a client anticheat or the raw input stream sees): mouse
  motion kinematics, click intervals, key timing.
- **Server-side observable** (what your server anticheat actually receives): rotation deltas
  per tick, position, hit/reach, attack cadence, response latency to visible events.

### 7.1 Candidate residual signatures

| Signal | Why a vision bot leaks it | Observable at |
|--------|---------------------------|---------------|
| Reaction-time distribution shape | Motor model's latency sampling is never exactly human | client + server (via response-to-event lag) |
| Aim-path velocity / jerk profile | Synthetic trajectories differ from hand kinematics | client (rich), server (subsampled per tick) |
| Fourier signature of mouse motion | Tremor/noise spectrum differs from human hand | client |
| Click-interval structure | Even humanized clicks lack true human refractory patterns | client + server (attack cadence) |
| Crit-timing regularity | RL/policy hits crit windows too consistently | server |
| Aim consistency vs. distance/occlusion | Bot degrades differently than humans near occlusion | server |
| Cross-session invariance | Humans vary day-to-day; a model is stationary | server (aggregate) |
| Target-switch latency | Perception re-acquisition has a characteristic delay | server |

### 7.2 Detector design

- Featurize each match into the signal set above.
- Model: start with gradient-boosted trees on engineered features (interpretable — you want to
  *know why* it flagged), then compare against a sequence model on raw rotation/input streams.
- Train on: human sessions (negative) vs. bot sessions at varying motor-fidelity levels
  (positive). The motor-fidelity sweep tells you the detection floor: *at what human-likeness
  does the bot become undetectable server-side?* That number is the whole point.
- Output: score + top contributing signals + which regime (client vs server) caught it.

### 7.3 Key research question to answer

Produce a curve: **motor-model fidelity (x) vs. server-side detectability (y)**. The
inflection point tells your anticheat where to invest — and whether server-side telemetry
alone is sufficient or whether you need client attestation.

---

## 8. macOS-first constraints (development target)

Development and initial testing target **macOS / Apple Silicon**. What that platform is good
and bad at directly shapes the phasing:

- **Good for:** dataset collection, BC training, perception, executor prototyping, the
  detector harness, RL *inference/eval*, and — crucially — the **fast `sim/` RL loop**, which
  is pure CPU and runs many times real-time.
- **Weak for:** large-scale RL training *throughput* against the real client. No CUDA (PyTorch
  uses the **MPS/Metal** backend; some ops fall back to CPU), and a single Minecraft client
  runs at ~real-time — parallelizing means many heavy JVM+render instances, and headless
  rendering on macOS is awkward.
- **Mitigations:** (1) the standalone `sim/` decouples RL from the client entirely; (2)
  privileged state-vector policy RL avoids the vision forward-pass; (3) if end-to-end pixel RL
  is ever needed at scale, rent a CUDA box for that phase only — do not architect around it now.

Executor backend on macOS: `CGEvent` for synthetic mouse/keyboard. (Realistic evasion research
may later want a Windows target where most cheats + client anticheat live — deferred.)

## 9. Real-time runtime budget

Per 30 Hz step (~33 ms): capture + perception + policy + executor must fit with margin.

- Capture: dxcam (Win) / mss can do <5 ms at downscaled res.
- Perception + policy: target <15 ms combined via ONNX Runtime (CUDA / DirectML) or
  TensorRT; on Mac, CoreML / Neural Engine.
- Executor dispatch: <1 ms.
- Keep models small; quantize (INT8) if needed. Measure end-to-end input-photon-to-action lag —
  it is itself a signature.

---

## 10. Phased roadmap

| Phase | Goal | Exit criterion |
|-------|------|----------------|
| **A. Fast-sim self-play** | Standalone `sim/` + PPO self-play sparring bot (privileged state, no vision, no client) | Trained policy beats the scripted baseline > 80% and beats older self-play checkpoints |
| **0. Instrumentation** | Recorder + ground-truth mod + sync | Clean aligned dataset for one session, verified drift < 1 tick |
| **1. Dataset** | Automated round segmentation, corpus build | 50+ hrs labeled, meta captured |
| **2. Perception** | Detector + state head | On-screen loc error and state F1 hit targets vs ground truth |
| **3. Policy (BC)** | Behavioral-cloned bot | Bot completes rounds vs. a scripted dummy from vision alone |
| **4. Policy (RL fine-tune)** | Port Phase-A self-play onto client/ground-truth env; BC→RL | Beats BC baseline; superhuman signatures logged for §7 |
| **5. Executor** | Human motor model | Aim/click kinematics pass a human-vs-bot blind stat test at low fidelity |
| **6. Blue-team** | Detector + fidelity/detectability curve | The §7.3 curve produced and documented |
| **B. Live vision bot** | End-to-end integration: pixels → action on a real client, run against your anticheat | A no-memory bot completes real duels on your server driven only by vision + synthetic input |

**Milestone A is complete** (the RL brain). Phase 0 is the current work. **Milestone B** is the
first end-to-end integration and the project's first anticheat-testable artifact — see §10.1.

### 10.1 Milestone B — live vision-to-action bot vs. a real anticheat

**Goal:** a bot that plays a live Minecraft Java client using *only* screen pixels in and
synthetic mouse/keyboard out — no memory reads, no packet injection — so it can be pointed at
your anticheat on your own server as a controlled red-team test.

Milestone B is not new theory; it is the **convergence** of existing phases into one real-time
loop:

```
   screen frames ──▶ [Perception, Phase 2] ──▶ state vector
                                                    │
                                                    ▼
                                       [Policy, Milestone A / Phase 3-4]
                                                    │
                                                    ▼
   synthetic HID ◀── [Executor, Phase 5] ◀──── action
        │
        ▼
   real Minecraft client on YOUR server ──▶ observed by YOUR anticheat
```

**Critical path to Milestone B:**

1. **Phase 0 (instrumentation)** — recorder + ground-truth mod. Gateway: perception can't be
   trained without labeled data, and the mod auto-labels it. *(current work)*
2. **Phase 2 (perception)** — pixels → state vector. The heaviest lift; the "visual" half.
3. **Policy** — reuse the Milestone-A brain as the initial decision layer; calibrate / fine-tune
   for the sim-to-real gap (or BC from real matches, Phase 3).
4. **Phase 5 (executor)** — action → human-like `CGEvent` mouse/keyboard on the real client.
5. **Integration** — wire the four into a ~15–30 ms real-time loop; run on your server.

**Reduced-scope early test:** to answer "does my anticheat flag a no-memory vision bot?"
*before* the RL brain is production-ready, perception + executor + even a simple heuristic
policy already exercise the anticheat against vision-based control. The RL brain makes the bot
*strong*; perception + executor make it *real* — and the anticheat mostly scrutinizes the latter
two. This is the smallest thing that produces a signal for the blue-team work (§7).

**Guardrail:** Milestone B runs on a server you control against your own anticheat — the
intended controlled red-team. It is not for public servers against non-consenting players (§0).

---

## 11. Open questions / decisions to revisit

- **Executor target OS:** dev/testing on macOS (decided). Realistic evasion research may later
  need Windows (where most cheats + client anticheat live) — deferred, not architected around.
- **Kit scope for v1:** pure sword 1v1 first; crystals/pots/pearls are much larger action
  spaces — defer.
- **End-to-end vs modular** perception: modular for v1 (decided); revisit VPT-style later.
- **Model conditioning** on sensitivity/FOV/GUI-scale vs. training across a spread — decide
  after first perception results.
- **How much server-side telemetry your anticheat actually logs today** — this bounds what
  §7 can realistically use; align the detector's feature set to your real observability.

---

## 12. Repository layout

Scaffolded so far (Milestone A), plus planned dirs:

```
BladeVision/
  docs/DESIGN.md              # this file
  sim/                        # [A] standalone sword-PvP simulator (CPU, no client)
    mechanics.py              #     1.9+ combat constants + hit/knockback/crit resolution
    arena.py                  #     two-player physics core (no RL dependency)
    gym_env.py                #     single-agent Gymnasium wrapper (opponent = frozen policy)
    opponents.py              #     scripted baseline + self-play checkpoint pool
  policy/                     # [A] training + eval
    selfplay.py               #     PPO self-play training loop (SB3)
    duel.py                   #     head-to-head eval between checkpoints
  requirements.txt            # [A]
  README.md                   # [A]
  recorder/                   # (planned) Python capture + input hook
  mod/                        # (planned) Fabric ground-truth mod (lab only)
  perception/                 # (planned) PyTorch models, training, ONNX export
  executor/                   # (planned) HID motor model (CGEvent on macOS)
  detector/                   # (planned) blue-team harness + signature features
  data/                       # (planned, gitignored) datasets
```
