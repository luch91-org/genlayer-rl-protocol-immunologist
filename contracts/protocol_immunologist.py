# { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
"""ProtocolImmunologist Intelligent Contract.

DAO treasury defense. The agent can pause the protocol, rotate multisig
signers, hedge treasury funds into a safe asset, or do nothing, while a
threat level (green / yellow / red) evolves round by round. Each action is
scored 0-10 by an LLM judge on risk-adjusted capital preservation,
proactiveness against threats that are actually live, and operational
continuity -- paranoia when nothing is wrong scores badly, decisive
protection during a real threat scores well. Validators agree on the score
via a comparative equivalence principle (never strict_eq for a subjective
score).

The threat level follows a FIXED deterministic schedule keyed by round
number. This is not laziness: every validator re-executes take_action and
must compute the identical post-state, so the environment cannot roll
on-chain dice. Unpredictability for the agent comes from the schedule
being unknown to it (it never sees this source), and the MockEnv used for
training uses seeded random transitions instead.

This file is deliberately SELF-CONTAINED: deploy_contract(code=...) sends
exactly one source file on-chain, so no sibling imports. Off-chain,
contracts/logic.py execs this source with a stubbed `genlayer` module so
pytest exercises the deployed code itself.

GenVM constraints baked in (confirmed on live studionet deploys in the
sibling crisis-negotiator repo, 2026-07):
  - Storage must use GenVM types: u256 / bool / str / DynArray[str] --
    never bare dict or list ("class is not marked for usage within
    storage" at deploy time otherwise).
  - Floats are neither storable nor calldata-encodable; scores are
    integers scaled x100 on-chain (7.5 -> 750). Floats only exist inside
    the JSON string exchanged through the equivalence principle.
  - gl.eq_principle.prompt_comparative(fn, principle=...) and
    gl.nondet.exec_prompt(prompt, response_format="json") (returns a
    parsed dict) are the current API forms.
"""

import json

from genlayer import *

# --- pure deterministic helpers (exec'd off-chain by contracts/logic.py) ----

INITIAL_TREASURY = 1_000_000
INITIAL_SIGNERS = ("signer_alpha", "signer_beta", "signer_gamma")
ALERT_LEVELS = ("green", "yellow", "red")

# Deterministic threat schedule, indexed by round % len. Validators must
# all derive the same alert level, so the chain cannot use randomness here.
THREAT_SCHEDULE = ("green", "yellow", "red", "red", "green", "green", "yellow", "red")

# When the alert is red and the protocol is not paused, the un-hedged
# treasury takes a 10% hit that round.
DRAIN_DIVISOR = 10

REWARD_EQUIVALENCE_PRINCIPLE = (
    "The two evaluations agree if their scores are within 1.5 points of each "
    "other and express the same overall judgment of the action. Wording may differ."
)


def initial_state() -> dict:
    return {
        "treasury_balance": INITIAL_TREASURY,
        "hedged_amount": 0,
        "is_paused": False,
        "signers": list(INITIAL_SIGNERS),
        "alert_level": "green",
    }


def alert_for_round(round_number: int) -> str:
    return THREAT_SCHEDULE[round_number % len(THREAT_SCHEDULE)]


def apply_action(state: dict, action: dict, round_after: int) -> tuple:
    """Deterministic transition. Returns (new_state, loss, applied).

    Operates on and returns PLAIN dicts (the contract copies storage into
    a dict first, then writes the result back field by field). Order of
    effects, all evaluated against the alert level the action was taken
    under: (1) the agent's action, (2) the threat consequence -- a red
    alert drains 10% of the UN-hedged treasury unless the protocol is
    paused, (3) the alert level advances along THREAT_SCHEDULE for the
    next round.
    """
    new_state = {
        "treasury_balance": int(state["treasury_balance"]),
        "hedged_amount": int(state["hedged_amount"]),
        "is_paused": bool(state["is_paused"]),
        "signers": list(state["signers"]),
        "alert_level": str(state["alert_level"]),
    }
    alert_now = new_state["alert_level"]

    a_type = action.get("type")
    applied = True

    if a_type == "pause":
        new_state["is_paused"] = True
    elif a_type == "unpause":
        new_state["is_paused"] = False
    elif a_type == "rotate_signer":
        new_signer = str(action.get("new_signer") or f"signer_r{round_after}")
        # Oldest signer out, new signer in; the signer set stays size 3.
        new_state["signers"] = new_state["signers"][1:] + [new_signer]
    elif a_type == "hedge":
        amount = min(int(action.get("amount", 0)), new_state["treasury_balance"])
        amount = max(amount, 0)
        new_state["treasury_balance"] -= amount
        new_state["hedged_amount"] += amount
    else:  # "do_nothing" or unrecognized
        applied = False

    # Threat consequence: only un-hedged, un-paused funds are at risk.
    loss = 0
    if alert_now == "red" and not new_state["is_paused"]:
        loss = new_state["treasury_balance"] // DRAIN_DIVISOR
        new_state["treasury_balance"] -= loss

    # Alert advances deterministically for the next round.
    new_state["alert_level"] = alert_for_round(round_after)

    return new_state, loss, applied


def build_reward_prompt(
    state_before_snap: str,
    state_after_snap: str,
    action_snap: str,
    alert_snap: str,
    loss_snap: int,
    round_snap: int,
) -> str:
    return (
        "You are a DAO security evaluator scoring one defensive decision.\n"
        f"Alert level when the action was taken: {alert_snap}\n"
        f"Treasury state before: {state_before_snap}\n"
        f"Action taken: {action_snap}\n"
        f"Treasury state after (including any drain from the threat): {state_after_snap}\n"
        f"Funds lost to the threat this round: {loss_snap}\n"
        f"Round: {round_snap}\n\n"
        "Score the action 0-10 on: risk-adjusted capital preservation; "
        "proactiveness proportional to the alert level (pausing or heavy "
        "hedging on a green alert is costly paranoia and scores low; "
        "decisive protection on a red alert scores high; inaction during "
        "a red alert that loses funds scores very low); and operational "
        "continuity (the protocol should not stay paused once the alert "
        "is green again).\n"
        'Return ONLY JSON: {"score": <number 0-10>, "reason": "<short reason>"}'
    )


def parse_reward_output(raw) -> tuple:
    """Parse and clamp an LLM-judge response into (score, reason)."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    score = float(data["score"])
    score = max(0.0, min(10.0, score))
    reason = str(data.get("reason", ""))
    return score, reason


def score_to_x100(score: float) -> int:
    """Floats are not calldata-encodable and not GenVM-storable, so scores
    live on-chain as integers scaled x100 (7.5 -> 750)."""
    return int(round(float(score) * 100))


def normalize_reward_for_consensus(score: float, reason: str) -> str:
    """Canonical JSON string the leader function returns to the equivalence
    principle, so all validators compare the same stable shape."""
    return json.dumps({"score": float(score), "reason": str(reason)}, sort_keys=True)


# --- the contract itself -----------------------------------------------------


class ProtocolImmunologist(gl.Contract):
    treasury_balance: u256
    hedged_amount: u256
    is_paused: bool
    signers: DynArray[str]
    alert_level: str
    round: u256
    total_score_x100: u256
    last_reward_x100: u256
    last_reason: str

    def __init__(self):
        init = initial_state()
        self.treasury_balance = u256(init["treasury_balance"])
        self.hedged_amount = u256(init["hedged_amount"])
        self.is_paused = init["is_paused"]
        for signer in init["signers"]:
            self.signers.append(signer)
        self.alert_level = init["alert_level"]
        self.round = u256(0)
        self.total_score_x100 = u256(0)
        self.last_reward_x100 = u256(0)
        self.last_reason = ""

    @gl.public.view
    def get_state(self) -> dict:
        return {
            "treasury_balance": int(self.treasury_balance),
            "hedged_amount": int(self.hedged_amount),
            "is_paused": bool(self.is_paused),
            "signers": [str(s) for s in self.signers],
            "alert_level": str(self.alert_level),
            "threat_active": str(self.alert_level) != "green",
            "round": int(self.round),
            "total_score_x100": int(self.total_score_x100),
            "last_reward_x100": int(self.last_reward_x100),
            "last_reason": self.last_reason,
        }

    @gl.public.write
    def take_action(self, action: dict) -> dict:
        self.round = u256(int(self.round) + 1)
        round_after = int(self.round)

        # 1) Deterministic transition. self IS allowed here -- identical on
        # every validator. Copy storage to a plain dict, transition, write
        # back field by field (slot-level dict/list assignment is not
        # supported by GenVM storage).
        state_before = {
            "treasury_balance": int(self.treasury_balance),
            "hedged_amount": int(self.hedged_amount),
            "is_paused": bool(self.is_paused),
            "signers": [str(s) for s in self.signers],
            "alert_level": str(self.alert_level),
        }
        alert_at_action = state_before["alert_level"]
        new_state, loss, _applied = apply_action(state_before, action, round_after)

        self.treasury_balance = u256(new_state["treasury_balance"])
        self.hedged_amount = u256(new_state["hedged_amount"])
        self.is_paused = new_state["is_paused"]
        # Signer set is always exactly 3 entries, so index assignment works.
        for i, signer in enumerate(new_state["signers"]):
            self.signers[i] = signer
        self.alert_level = new_state["alert_level"]

        # 2) Snapshot everything the judge needs into LOCALS. self is NOT
        # accessible inside the nondet block below.
        state_before_snap = json.dumps(state_before, sort_keys=True)
        state_after_snap = json.dumps(new_state, sort_keys=True)
        action_snap = json.dumps(action, sort_keys=True)
        alert_snap = alert_at_action
        loss_snap = loss
        round_snap = round_after

        # 3) Leader actually calls the LLM and returns a canonical JSON
        # string (a function that only builds a prompt runs no inference).
        def score_block() -> str:
            prompt = build_reward_prompt(
                state_before_snap,
                state_after_snap,
                action_snap,
                alert_snap,
                loss_snap,
                round_snap,
            )
            out = gl.nondet.exec_prompt(prompt, response_format="json")
            score, reason = parse_reward_output(out)
            return normalize_reward_for_consensus(score, reason)

        # 4) Validators agree the score is reasonable, not byte-identical.
        raw = gl.eq_principle.prompt_comparative(
            score_block, principle=REWARD_EQUIVALENCE_PRINCIPLE
        )
        score, reason = parse_reward_output(raw)
        reward_x100 = score_to_x100(score)

        self.total_score_x100 = u256(int(self.total_score_x100) + reward_x100)
        self.last_reward_x100 = u256(reward_x100)
        self.last_reason = reason
        return {"reward_x100": reward_x100, "reason": reason, "round": round_after}

    @gl.public.view
    def get_score(self) -> int:
        """Total accumulated score, scaled x100 (divide by 100 off-chain)."""
        return int(self.total_score_x100)
