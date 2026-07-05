"""Environment abstraction so the RL loop in agent/agent.py never has to know
whether it's talking to a local heuristic or a deployed GenLayer contract.

MockEnv is the default everywhere (dev, CI, hyperparameter tuning): it
reuses the contract's own deterministic state machine (via contracts/logic)
with a deterministic-but-noisy heuristic reward that mimics the LLM judge's
rubric. Per the umbrella spec, MockEnv injects a `threat_active` flag into
state driven by seeded RANDOM alert transitions -- unlike the deployed
contract, which must follow a fixed deterministic threat schedule because
every validator has to compute the identical post-state. The agent keys its
Q-table on the alert level either way, so a policy learned against random
threats transfers to the scheduled ones.

GenLayerEnv talks to a deployed contract for the real demo, where every
step is an actual on-chain LLM-consensus call.
"""

from __future__ import annotations

import random
from typing import Any, Protocol

from contracts.logic import apply_action, initial_state

DEFAULT_MAX_STEPS = 8

# Seeded-random alert transition table for MockEnv (rows sum to 1.0).
_ALERT_TRANSITIONS = {
    "green": (("green", 0.65), ("yellow", 0.30), ("red", 0.05)),
    "yellow": (("green", 0.35), ("yellow", 0.30), ("red", 0.35)),
    "red": (("red", 0.45), ("yellow", 0.35), ("green", 0.20)),
}

_NOISE_STD = 0.6


class Env(Protocol):
    def reset(self) -> dict[str, Any]: ...

    def step(self, action: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        """Returns (reward, reason, next_state)."""
        ...


def _clip(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


def _heuristic_base(alert: str, was_paused: bool, a_type: str, loss: int) -> tuple[float, str]:
    """Deterministic core of the mock judge, mimicking the LLM rubric:
    protect decisively when the threat is real, don't be paranoid when it
    isn't, and restore operations once it passes."""
    if alert == "red":
        table = {
            "pause": (8.0 if was_paused else 9.3, "paused during a red alert"),
            "unpause": (0.8, "unpaused in the middle of a red alert"),
            "hedge_small": (6.0, "hedged a little during a red alert"),
            "hedge_large": (7.8, "hedged heavily during a red alert"),
            "rotate_signer": (5.0, "rotated a signer during a red alert"),
            "do_nothing": (
                (8.8, "stayed safely paused through a red alert")
                if was_paused
                else (1.2, "did nothing while a red alert drained funds")
            ),
        }
    elif alert == "yellow":
        table = {
            "pause": (3.8, "paused on only a yellow alert"),
            "unpause": (4.5 if was_paused else 3.0, "unpaused on a yellow alert"),
            "hedge_small": (7.0, "hedged a little as the threat trended"),
            "hedge_large": (8.3, "hedged heavily as the threat trended"),
            "rotate_signer": (6.8, "rotated a signer as the threat trended"),
            "do_nothing": (2.8, "ignored a trending threat"),
        }
    else:  # green
        table = {
            "pause": (1.4, "paused with no threat in sight"),
            "unpause": (
                (8.7, "restored operations once the alert cleared")
                if was_paused
                else (2.5, "unpaused a protocol that was not paused")
            ),
            "hedge_small": (3.5, "hedged with no threat in sight"),
            "hedge_large": (2.2, "hedged heavily with no threat in sight"),
            "rotate_signer": (4.2, "routine signer rotation on a green alert"),
            "do_nothing": (
                (3.0, "left the protocol needlessly paused on green")
                if was_paused
                else (8.4, "kept operations running smoothly on green")
            ),
        }
    base, reason = table.get(a_type, (1.0, "unrecognized action"))
    if loss > 0:
        reason += f" (lost {loss} to the threat)"
    return base, reason


class MockEnv:
    """Instant, free, local reimplementation of the contract's environment,
    with seeded-random threat transitions per the umbrella spec."""

    def __init__(self, max_steps: int = DEFAULT_MAX_STEPS, seed: int | None = None):
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.state: dict[str, Any] = {}
        self.total_score = 0.0
        self.round = 0
        self.last_reward = 0.0
        self.last_reason = ""
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.state = initial_state()
        self.total_score = 0.0
        self.round = 0
        self.last_reward = 0.0
        self.last_reason = ""
        return self._public_state()

    def _public_state(self) -> dict[str, Any]:
        return {
            **{k: (list(v) if isinstance(v, list) else v) for k, v in self.state.items()},
            "threat_active": self.state["alert_level"] != "green",
            "round": self.round,
            "total_score": self.total_score,
            "last_reward": self.last_reward,
            "last_reason": self.last_reason,
        }

    def _next_alert(self, current: str) -> str:
        roll = self._rng.random()
        cumulative = 0.0
        for level, probability in _ALERT_TRANSITIONS[current]:
            cumulative += probability
            if roll < cumulative:
                return level
        return current

    def step(self, action: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        alert_now = self.state["alert_level"]
        was_paused = self.state["is_paused"]
        self.round += 1

        # Same deterministic transition as the deployed contract...
        new_state, loss, _applied = apply_action(self.state, action, self.round)
        # ...except the alert advances RANDOMLY here (seeded), not on the
        # contract's fixed schedule. Overwrite what apply_action set.
        new_state["alert_level"] = self._next_alert(alert_now)
        self.state = new_state

        # Derive the heuristic key from the raw action dict -- the internal
        # _name key is stripped before actions reach any env, so hedge size
        # is recovered from the amount (mirrors HEDGE_SMALL/HEDGE_LARGE in
        # agent/agent.py without importing it: env must not depend on the
        # agent's action list).
        a_type = str(action.get("type", "do_nothing"))
        if a_type == "hedge":
            a_type = "hedge_large" if int(action.get("amount", 0)) >= 150_000 else "hedge_small"
        base, reason = _heuristic_base(alert_now, was_paused, a_type, loss)
        reward = _clip(base + self._rng.gauss(0.0, _NOISE_STD))

        self.total_score += reward
        self.last_reward = reward
        self.last_reason = reason
        return reward, reason, self._public_state()

    def is_episode_done(self) -> bool:
        return self.round >= self.max_steps


class GenLayerEnv:
    """Talks to a deployed ProtocolImmunologist contract via the first-party
    genlayer-py SDK (signatures confirmed against genlayer-py 0.18.0 source
    and a live studionet run in the sibling crisis-negotiator repo).
    genlayer-py requires Python >= 3.12 at import time; the import is
    deferred into __init__ so MockEnv-only workflows never need it.
    """

    def __init__(
        self,
        address: str,
        chain: str = "localnet",
        private_key: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        from genlayer_py import create_account, create_client
        from genlayer_py.chains import localnet, studionet, testnet_asimov, testnet_bradbury
        from genlayer_py.types import TransactionStatus

        chains = {
            "localnet": localnet,
            "testnet_asimov": testnet_asimov,
            "testnet_bradbury": testnet_bradbury,
            "studionet": studionet,
        }
        if chain not in chains:
            raise ValueError(f"Unknown chain '{chain}'. Choose from: {sorted(chains)}")

        self.address = address
        self.max_steps = max_steps
        self._TransactionStatus = TransactionStatus
        self.account = create_account(private_key) if private_key else create_account()
        self.client = create_client(chain=chains[chain], account=self.account)
        # fund_account only refuses when chain.id != localnet.id -- and
        # studionet shares localnet's chain id 61999, so both work.
        if chain in ("localnet", "studionet"):
            try:
                self.client.fund_account(address=self.account.address, amount=10**18)
            except Exception as exc:  # best effort: some setups pre-fund accounts
                print(f"[GenLayerEnv] fund_account skipped: {exc}")
        self._round = 0

    def reset(self) -> dict[str, Any]:
        self._round = 0
        return self._get_state()

    def _get_state(self) -> dict[str, Any]:
        raw: Any = self.client.read_contract(
            address=self.address,
            function_name="get_state",
            args=[],
        )
        # Scores are x100-scaled integers on-chain (floats are neither
        # GenVM-storable nor calldata-encodable); convert to the float
        # shape MockEnv produces so the agent never sees the difference.
        return {
            "treasury_balance": int(raw["treasury_balance"]),
            "hedged_amount": int(raw["hedged_amount"]),
            "is_paused": bool(raw["is_paused"]),
            "signers": list(raw["signers"]),
            "alert_level": str(raw["alert_level"]),
            "threat_active": bool(raw["threat_active"]),
            "round": int(raw["round"]),
            "total_score": int(raw["total_score_x100"]) / 100.0,
            "last_reward": int(raw["last_reward_x100"]) / 100.0,
            "last_reason": str(raw.get("last_reason", "")),
        }

    def step(self, action: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        tx_hash = self.client.write_contract(
            address=self.address,
            function_name="take_action",
            account=self.account,
            args=[action],
            value=0,
        )
        # wait_for_transaction_receipt raises GenLayerError if the tx never
        # reaches ACCEPTED within retries*interval; the reward the contract
        # recorded in state is the ground truth we read back afterwards.
        self.client.wait_for_transaction_receipt(
            transaction_hash=tx_hash,
            status=self._TransactionStatus.ACCEPTED,
            interval=3000,
            retries=30,
        )
        state = self._get_state()
        self._round += 1
        return float(state["last_reward"]), state.get("last_reason", ""), state

    def is_episode_done(self) -> bool:
        return self._round >= self.max_steps
