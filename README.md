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
  selfplay.py    # PPO self-play training (Stable-Baselines3)
  duel.py        # head-to-head eval between checkpoints
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
python -m policy.selfplay --generations 20 --steps-per-gen 200000 --out runs/selfplay
```

Generations 0–1 warm up against a scripted baseline; later generations play league self-play
against a rolling pool of frozen past checkpoints. Win rate vs. the baseline is printed each
generation.

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
