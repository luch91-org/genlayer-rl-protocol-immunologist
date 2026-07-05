# genlayer-rl-protocol-immunologist

Domain-scoped build guide. Inherits the shared engineering spec from the
[GenLayer RL Agent Autonomy](https://github.com/luch91-org)
umbrella CLAUDE.md - read that first. This file only covers what's specific
to this domain.

## Domain

DAO treasury defense. State = treasury balance, hedged reserve, pause
flag, 3-signer multisig set, and an alert level in `{green, yellow, red}`.
Actions = `pause`, `unpause`, `rotate_signer(new_signer?)`,
`hedge(amount)`, `do_nothing`. A red alert drains 10% of the un-hedged
treasury each round unless the protocol is paused. The agent learns to
protect capital when a threat is actually live, and NOT to protect it
when nothing is wrong - the judge explicitly punishes paranoia on green
and inaction on red.

## Where things live

- `contracts/protocol_immunologist.py` - the `gl.Contract`, and the
  single source of truth for ALL contract logic. Must stay fully
  self-contained (single-file deployment; regression guard in tests).
- `contracts/logic.py` - no logic of its own; execs the contract source
  with a stubbed `genlayer` module and re-exports the pure helpers, so
  pytest and MockEnv exercise the deployed code itself. New deterministic
  behavior goes in the contract as a module-level function, then gets
  re-exported in logic.py's explicit list.
- `agent/env.py` - MockEnv reuses the contract's `apply_action` but
  overrides the alert transition with seeded RANDOM transitions (per the
  umbrella spec's `threat_active` requirement). The deployed contract
  instead follows the fixed `THREAT_SCHEDULE` - validators must all
  compute the identical post-state, so the chain cannot roll dice. Keep
  that split intact.
- `agent/agent.py` - `serialize_state()` collapses to
  `(alert_level, is_paused, hedge_bucket, treasury_bucket)`, ≤36 states.
  Actions carry an internal `_name` key (hedge_small vs hedge_large);
  `strip_action()` removes underscore-keys before anything is sent
  on-chain - don't send internal keys to the contract.

## Non-negotiable GenLayer rules for this contract

(Full rationale in the umbrella CLAUDE.md.)

- Non-deterministic calls only inside the inner `score_block()` passed to
  `gl.eq_principle.prompt_comparative(fn, principle=...)`; never in the
  method body; never reference `self` inside it - snapshot to locals
  first.
- Never `strict_eq` for the subjective score; the stated tolerance lives
  in `REWARD_EQUIVALENCE_PRINCIPLE` (1.5 points).
- Storage must be GenVM types: `u256`, `bool`, `str`, `DynArray[str]` - 
  never bare `dict`/`list` (deploy fails with "class is not marked for
  usage within storage"; confirmed live). Populate collections element by
  element; the signer set stays exactly size 3 so index-assignment
  write-back works.
- No floats on-chain: scores are ×100-scaled integers (`*_x100` fields);
  `GenLayerEnv` converts back. `calldata.encode(1.5)` raises - confirmed.
- The threat schedule must stay deterministic on-chain. If you want
  richer threat dynamics, derive them from the round number or other
  consensus state - never from randomness or wall-clock time.

## Success bar for this repo

500 mock episodes climb from roughly 3-4 to 8+ per-step rolling average
(episode rewards are summed over `--max-steps`, default 8);
`agent/q_table.json`, `docs/learning_curve.png`, and `logs/training.txt`
exist afterward; and a short live run against a deployed contract shows
climbing episode rewards. The verified live deployment address and log
are recorded in the README.
