# Codenames LLM + RL

Two-team [Codenames](https://en.wikipedia.org/wiki/Codenames_(board_game)) self-play in Python: a small game engine, JSON-speaking LLM policies (spymaster + guesser), rollout export, **supervised fine-tuning (SFT)** warm-start, and **GRPO** (group relative policy optimization) training with a Hugging Face causal LM.

## Features

- **Game** (`game/`): `SelfPlayCodenamesGame` — board, turns, guesses, rewards, win/loss.
- **Policies** (`bots/`): prompt builders, structured JSON actions, `LLM*` adapters and `Random*` baselines.
- **LLM** (`llm/`): `TextGenerationBackend` protocol; **Hugging Face** implementation for local `generate` + training on the same weights.
- **Rollouts** (`runner/`): self-play episodes → `TrajectoryStep` / `EpisodeRecord`, discounted returns, JSONL export.
- **Clue validation** (`bots/spymaster/validation.py`): enforces legal Codenames clues (single alphabetic word, no substring overlap with board words, no repeats), with retry-and-feedback in the LLM spymaster and a forfeit penalty in the runner.
- **Evaluation** (`runner/evaluate.py`): head-to-head matchups with side-swapping; reports win rate, guess accuracy, assassin rate, illegal-clue rate, and clue diversity.
- **Training** (`training/`): **SFT** (filtered imitation on completions) and **GRPO** (advantage-weighted log-probs + KL vs a reference model), with token-level log-probs in `log_probs.py`.
- **Scripts**: collect JSONL without a GPU, play a game in the terminal, evaluate one bot against another, run `train.py sft|grpo`.
- **Tests** (`tests/`): engine, bots, clue validation, runner, evaluation, utils.

## Requirements

- **Python ≥ 3.11**
- **Core** (rollouts, tests): `numpy` (see `pyproject.toml`).
- **Training** (SFT / GRPO, HF backend): install optional **`train`** extras (`torch`, `transformers`, etc.).

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -e ".[train]"   # core + training stack; use pip install -e ".[dev]" for pytest
```

Editable install expects package discovery to ignore non-package dirs (`artifacts/`, `scripts/`, `tests/`); see `pyproject.toml`.

## Quick start

**Rollouts only (no GPU, random policies):**

```bash
python scripts/collect_rollouts.py --episodes 8 --output artifacts/rollouts.jsonl
```

**Tests:**

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
```

**SFT warm-start** (example: small model, CPU-friendly smoke run):

```bash
python scripts/train.py sft --model gpt2 --role guesser --episodes 64 --epochs 3 --output artifacts/sft
```

**GRPO** (example: load a model path, train guesser):

```bash
python scripts/train.py grpo --model artifacts/sft --role guesser --steps 100 --group-size 8 --output artifacts/grpo_ckpt
```

Use `python scripts/train.py --help` and `sft` / `grpo` subcommands for full flags (`--word-file`, board sizes, KL coefficient, etc.).

**Evaluate** a bot head-to-head (random vs random is a ~50/50 harness sanity check; point `--candidate-model` at a checkpoint to measure training):

```bash
python scripts/evaluate.py --games 100                          # sanity check
python scripts/evaluate.py --candidate-model artifacts/grpo_ckpt --baseline-model gpt2 --games 100
```

## Project layout

| Path | Role |
|------|------|
| `game/engine.py` | Game rules and per-guess rewards (`SelfPlayRewardConfig` in `game/config.py`). |
| `bots/spymaster/`, `bots/guesser/` | Observations, prompts, `LLM*` / `Random*` policies. |
| `llm/backends.py`, `llm/providers/huggingface.py` | Backend protocol and HF `generate` + shared `model` / `tokenizer` for training. |
| `runner/play.py` | `collect_self_play_episode(s)`, outcome bonuses, discounted returns. |
| `runner/trajectory.py` | `TrajectoryStep`, `EpisodeRecord`. |
| `runner/export.py` | `episode_to_jsonl_records`, SFT-style records, `write_jsonl`. |
| `runner/evaluate.py` | `evaluate_matchup`, `SideMetrics` — head-to-head win rate and diagnostics. |
| `bots/spymaster/validation.py` | `validate_clue` — legal-clue enforcement. |
| `training/sft.py` | Collect random rollouts → filter by `discounted_return` → CE on completions. |
| `training/grpo.py` | Each step: `G` episodes → normalize returns → weighted log-prob + KL. |
| `training/log_probs.py` | `encode_prompt_and_completion`, `completion_log_probs`. |
| `scripts/` | `collect_rollouts.py`, `train.py`, `play_game.py`, `evaluate.py`. |
| `wordlist-eng.txt` | Default word pool for boards. |

## Terminology

| Term | Meaning |
|------|--------|
| **Episode** | One full game until win, loss, or `max_turns`. |
| **Trajectory step** | One LLM call: one spymaster (clue) **or** one guesser (guess). One training row in export. |
| **Game turn** | One team’s clue **plus** their guessing sub-turn; contains **multiple** trajectory steps. |
| **SFT epoch** | One pass over the **fixed** dataset built from `n` collection episodes. |
| **GRPO step** | New batch of `G` episodes → advantages → one optimizer update (no “epoch” over fixed data). |

## Training notes

- **SFT** keeps steps whose `discounted_return >= minimum_return` (default `0`), then minimizes negative log-likelihood on the completion **without** reward-weighted loss; reward only gates **which** rows train.
- **GRPO** uses **all** steps for the trained role(s) in the batch, normalizes `discounted_return` across those records for advantages, and adds a KL penalty against a frozen copy of the initial model.
- The other role during training is often a **random** policy so one role is trained at a time (`--role guesser` / `spymaster` / `both`).

## Extending

- Implement `TextGenerationBackend` in `llm/backends.py` for non–Hugging Face providers; pass it into `LLMSpymasterPolicy` / `LLMGuesserPolicy`.
- **Batching** and **parallel envs** are left to the user (scaffold-focused).

## License

MIT — see the [OSI text](https://opensource.org/licenses/MIT) for the full license terms.
