# BladeVision

**A vision-based sword-PvP AI for Minecraft Java Edition, built as anticheat red-team / blue-team research.**

BladeVision studies the emerging class of game cheats that read **only the screen** and emit
**synthetic mouse/keyboard input** — touching no game memory and injecting no packets, so
traditional anticheat (memory scanning, packet/rotation analysis) is blind to them. We build
that cheat in a controlled lab in order to answer the question that matters to an anticheat
operator:

> When a cheat reads no memory and injects no packets, what detectable signal is left, and how
> much of it survives to the server side where the anticheat can actually observe it?

The working bot is the means; the **detector + residual-signature catalog** is the deliverable.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full architecture and roadmap.

## Research guardrails

- **Isolated environment only** — training/eval on a private server you control (bot-vs-bot,
  you-vs-bot). Not for deployment against unwitting players on public servers.
- **The ground-truth mod is lab instrumentation, never shipped** — a separate artifact from any
  executor, used only for data collection and detector training.
- **Blue-team parity** — every attack capability is matched by analysis of the signature it leaves.

## Milestone A — fast-sim self-play (this scaffold)

Before any Minecraft client or vision work, we de-risk the RL loop on a standalone, CPU-only
sword-PvP simulator. It models 1.9+ combat mechanics (reach, attack-cooldown charge, sprint
knockback + sprint-reset, jump crits) and runs many times faster than real-time — including on
Apple Silicon, with no CUDA.

```
sim/
  mechanics.py   # combat constants + pure hit/knockback/crit resolution
  arena.py       # two-player physics core (immutable state snapshots, no RL dep)
  gym_env.py     # single-agent Gymnasium wrapper (opponent = frozen policy)
  opponents.py   # scripted baseline + self-play checkpoint pool (league)
policy/
  selfplay.py    # PPO self-play training (Stable-Baselines3), resumable daemon
  duel.py        # head-to-head eval between checkpoints
dashboard/
  server.py      # stdlib web server exposing /api/progress
  index.html     # live auto-refreshing training dashboard (charts + table)
tests/
  test_mechanics.py  # dependency-free sanity checks on the sim
```

### Setup

```bash
cd BladeVision
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Sanity-check the simulator (no RL deps needed)

```bash
python -m pytest tests/ -q          # or: python tests/test_mechanics.py
```

### Train a self-play sword bot

```bash
python -m policy.selfplay --generations 20 --steps-per-gen 150000 --out runs/selfplay
```

Warmup generations (default first 3) train against the **scripted duelist baseline** — a
competent sword bot that circle-strafes and jump-crits. Later generations play league
self-play against a rolling pool of frozen past checkpoints. No human input is ever required.

**Unattended daemon:** run it hands-off; it checkpoints and logs every generation, so a
stop/crash loses at most the in-progress one.

```bash
# run indefinitely in the background; resume later from the newest checkpoint
nohup python -m policy.selfplay --forever --steps-per-gen 150000 --out runs/selfplay > train.log 2>&1 &
python -m policy.selfplay --resume --forever --out runs/selfplay   # continue after a stop
```

Two metrics are logged per generation to `runs/<name>/progress.jsonl` (+ latest in
`status.json`): **win rate vs. the fixed baseline** (a yardstick that saturates near 100% once
competent) and **win rate vs. gen 0** (the earliest self — a non-saturating signal of genuine
self-play improvement).

### Live dashboard

```bash
python -m dashboard.server --runs runs/selfplay --port 8765
# open http://127.0.0.1:8765
```

Auto-refreshing status cards, improvement charts (vs-baseline, vs-gen0, throughput), and a
recent-generation table. Pure stdlib — no extra dependencies.

### Evaluate

```bash
python -m policy.duel runs/selfplay/final.zip --vs scripted --episodes 200
python -m policy.duel runs/selfplay/gen_019.zip --vs runs/selfplay/gen_000.zip
```

**Milestone A exit criterion:** the trained policy beats the scripted baseline > 80% and beats
older self-play checkpoints — proving the RL loop, reward design, and self-play league before
we attach any client, vision, or human-motor executor.

## Status

Milestone A scaffolded. Later phases (recorder, Fabric ground-truth mod, perception, executor,
detector) are described in `docs/DESIGN.md` and not yet implemented.
