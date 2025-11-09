# SAAMC
Self-Aware Affect-Modulated Controller (SAAMC): a PyTorch + Minigrid agent combining affect (valence/arousal/novelty), a Global Workspace gate, metacognition, and a safety kernel with PPO. 
# SAAMC — Self-Aware Affect-Modulated Controller

> A small, transparent research agent that uses **affect dynamics** (valence, arousal, novelty),
> a **Global Workspace** broadcast, **metacognitive uncertainty**, and a **Safety Kernel**—trained
> with PPO on MiniGrid. Includes ablations, logs, and plotting helpers.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-red.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="PRs welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
</p>

## Why this matters
SAAMC treats **affect as a control signal** rather than a side log. A learned **Global Workspace (GW) gate** decides which internal reps (perception, memory, affect) get
system-wide access. A **metacognitive head** estimates uncertainty, and a simple **Safety Kernel** throttles risky actions when *arousal ↑* and *uncertainty ↑*.
You can show: (i) better sample efficiency vs ablations, (ii) interpretable GW weights, and (iii) fewer catastrophes under distribution shift.

---

## Repo contents
- `saamc_research_build.py` — runnable research build (PyTorch + Gymnasium + Minigrid)  
- `SAAMC_breakthrough_plan.md` — concise milestones, metrics, and figures to target  
- `saamc_logs/` — created at runtime with CSV logs and plots (ignored by Git by default)

---

## Install

```bash
# Create & activate a venv if you like
pip install gymnasium minigrid torch torchvision einops matplotlib pandas
# (For GPU-specific Torch wheels, follow the official PyTorch install page.)

