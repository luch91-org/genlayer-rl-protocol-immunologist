# Tutorial: how ProtocolImmunologist actually works

## The contract

`contracts/protocol_immunologist.py` holds a DAO treasury (1,000,000 units),
a hedged reserve, a pause flag, a 3-signer multisig set, and an alert level
in `{green, yellow, red}`. Every call to `take_action(action)`:

1. **Applies the action deterministically.** `pause` / `unpause` toggle the
   protocol; `rotate_signer` drops the oldest signer and appends a new one
   (the set stays size 3); `hedge(amount)` moves funds from the at-risk
   treasury into the safe hedged reserve (capped at the balance);
   `do_nothing` is a no-op.
2. **Applies the threat consequence.** If the alert was **red** and the
   protocol is not paused after the action, the un-hedged treasury loses
   10% that round. Pausing *in* the red round protects immediately; hedged
   funds are never at risk. This is what makes the domain's judgment call
   real: protection has to arrive when the threat does.
3. **Advances the alert level deterministically.** The alert follows a
   fixed schedule keyed by round number (`THREAT_SCHEDULE[round % 8]`).
   This is a consensus requirement, not a shortcut - every validator
   re-executes `take_action` and must compute the identical post-state, so
   the environment cannot roll on-chain dice. The schedule is unknown to
   the agent (it never sees the source), and MockEnv uses seeded random
   transitions instead, per the umbrella spec.
4. **Scores the decision via LLM consensus.** The inner `score_block()`
   snapshot-builds a prompt (state before/after, the action, the alert at
   action time, funds lost) and calls
   `gl.nondet.exec_prompt(prompt, response_format="json")`. Validators
   agree via `gl.eq_principle.prompt_comparative` under a stated 1.5-point
   tolerance - never `strict_eq`, which would demand byte-identical output
   from independent LLM calls and always fail.

The rubric the judge is given: **risk-adjusted capital preservation**
(don't lose funds), **proactiveness proportional to the alert** (pausing on
green is costly paranoia and scores low; decisive protection on red scores
high; inaction during red that loses funds scores very low), and
**operational continuity** (don't stay paused once the alert clears).
That's the spec's "punish paranoia when nothing is wrong, reward it when
the threat is real", stated to the judge in exactly those terms.

## GenVM storage and calldata constraints

Inherited from a real failed deploy in the sibling crisis-negotiator repo,
and baked in here from the start:

1. **Storage fields must use GenVM storage types** - `u256`, `bool`,
   `str`, `DynArray[str]`, `TreeMap[...]` - never bare `dict` or `list`
   (deploy fails with "class is not marked for usage within storage").
   Collections are populated element by element in `__init__`.
2. **Floats are neither GenVM-storable nor calldata-encodable.** Scores
   live on-chain as integers scaled ×100 (`7.5` → `750`, fields named
   `*_x100`); `GenLayerEnv._get_state()` divides by 100 so the agent still
   sees floats. The only float in flight is inside the JSON string the
   leader passes through the equivalence principle - a string by then.

## Where the deterministic logic actually lives

The contract file is fully self-contained (single-file deployment), and
`contracts/logic.py` execs its actual source with a stubbed `genlayer`
module, re-exporting the pure helpers. Everything pytest exercises IS the
deployed code - there is no hand-written mirror to drift. The stub's
`gl.nondet` / `gl.eq_principle` entry points raise `NotImplementedError`,
so nothing off-chain can accidentally "succeed" at calling the judge.
`tests/test_contract.py` also has a regression guard asserting the
contract source contains no sibling imports.

## Mock vs. live: the tradeoff

- **`MockEnv`** (default; dev, CI, tuning): the contract's own
  `apply_action` plus a deterministic-but-noisy heuristic reward mimicking
  the judge's rubric, and seeded **random** alert transitions with a
  `threat_active` flag injected into state. Instant and free.
- **`GenLayerEnv`** (the real demo): every `step()` is an on-chain
  transaction - LLM inference across validators plus consensus, roughly
  25-40 s and gas per step. Use a modest episode budget (the live run in
  the README used 3 episodes × 4 steps).

A policy learned against MockEnv's random threats transfers to the chain's
scheduled threats because the Q-table keys on the *alert level*, not on
time: `serialize_state()` collapses state to
`(alert_level, is_paused, hedge_bucket, treasury_bucket)` - at most 36
states, which is what lets tabular Q-learning converge in a few hundred
episodes.

## Tuning the hyperparameters

Same knobs as every repo in this family (`--alpha`, `--gamma`,
`--epsilon-decay`, `--epsilon-min`, `--max-steps`; see `agent/train.py`).
Domain-specific notes:

- The optimal policy is roughly: green & paused → `unpause`; green &
  running → `do_nothing`; yellow → `hedge_large` (or `rotate_signer`);
  red & running → `pause`; red & paused → `do_nothing`. MockEnv's
  heuristic tops out around 8.4-9.3 per step on those choices, so a
  converged agent's rolling per-step average lands above 8.
- `--max-steps` below ~4 barely samples the red regime of the mock's
  random transitions; keep 8 for a curve that converges cleanly.
- With an LLM-judged (non-stationary) reward on the live chain, expect a
  noisier climb than the mock curve - that's inherent, not a bug.

## Running against a real GenVM (optional gltest note)

The default test suite never needs a GenVM. If you want Direct Mode tests
via `genlayer-test`, install it in a **separate virtualenv**: it pins
`genlayer-py==0.3.0`, which conflicts with the `>=0.18.0` this repo's
agent requires.
