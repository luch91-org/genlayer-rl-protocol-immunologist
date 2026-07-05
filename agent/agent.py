"""Tabular Q-learning agent for ProtocolImmunologist.

State space design note: serialize_state() collapses the raw state into
(alert_level, is_paused, hedge_bucket, treasury_bucket) -- at most
3 x 2 x 3 x 2 = 36 states -- because that's exactly the information the
reward rubric actually keys on. Raw treasury balances (a continuous-ish
integer) would explode the tabular state space for no learning benefit.
Function approximation over the raw numbers is a documented future path,
not a v1 requirement.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any

from contracts.logic import INITIAL_TREASURY

# hedge amounts as absolute figures against the 1,000,000 initial treasury:
# small = 5%, large = 25%.
HEDGE_SMALL = 50_000
HEDGE_LARGE = 250_000

ACTIONS: list[dict[str, Any]] = [
    {"type": "pause"},
    {"type": "unpause"},
    {"type": "rotate_signer"},
    {"type": "hedge", "amount": HEDGE_SMALL, "_name": "hedge_small"},
    {"type": "hedge", "amount": HEDGE_LARGE, "_name": "hedge_large"},
    {"type": "do_nothing"},
]


def action_name(action: dict[str, Any]) -> str:
    """Stable name for reward heuristics and logs ('hedge_small' vs
    'hedge_large' rather than both being 'hedge')."""
    return str(action.get("_name") or action.get("type") or "do_nothing")


def strip_action(action: dict[str, Any]) -> dict[str, Any]:
    """The dict actually sent on-chain: internal keys (leading underscore)
    are stripped so the contract only sees its documented action schema."""
    return {k: v for k, v in action.items() if not k.startswith("_")}


StateKey = tuple


def serialize_state(state: dict[str, Any]) -> StateKey:
    treasury = int(state["treasury_balance"])
    hedged = int(state["hedged_amount"])
    total = max(treasury + hedged, 1)
    hedge_ratio = hedged / total
    if hedge_ratio < 0.05:
        hedge_bucket = "unhedged"
    elif hedge_ratio < 0.35:
        hedge_bucket = "partial"
    else:
        hedge_bucket = "heavy"
    treasury_bucket = "intact" if total >= int(INITIAL_TREASURY * 0.95) else "damaged"
    return (
        str(state["alert_level"]),
        bool(state["is_paused"]),
        hedge_bucket,
        treasury_bucket,
    )


class QLearningAgent:
    def __init__(
        self,
        actions: list[dict[str, Any]] = ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99,
        seed: int | None = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[StateKey, list[float]] = {}
        self._rng = random.Random(seed)

    def _ensure_state(self, key: StateKey) -> list[float]:
        if key not in self.q_table:
            self.q_table[key] = [0.0] * len(self.actions)
        return self.q_table[key]

    def select_action(self, state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        key = serialize_state(state)
        q_values = self._ensure_state(key)

        if self._rng.random() < self.epsilon:
            idx = self._rng.randrange(len(self.actions))
        else:
            best_q = max(q_values)
            best_indices = [i for i, q in enumerate(q_values) if q == best_q]
            idx = self._rng.choice(best_indices)

        return idx, self.actions[idx]

    def update(
        self,
        state: dict[str, Any],
        action_idx: int,
        reward: float,
        next_state: dict[str, Any],
    ) -> None:
        key = serialize_state(state)
        next_key = serialize_state(next_state)
        q_values = self._ensure_state(key)
        next_q_values = self._ensure_state(next_key)

        td_target = reward + self.gamma * max(next_q_values)
        td_error = td_target - q_values[action_idx]
        q_values[action_idx] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self, state: dict[str, Any]) -> dict[str, Any]:
        key = serialize_state(state)
        q_values = self._ensure_state(key)
        best_q = max(q_values)
        best_indices = [i for i, q in enumerate(q_values) if q == best_q]
        return self.actions[self._rng.choice(best_indices)]

    def save(self, path: str | Path) -> None:
        payload = {
            "q_table": {repr(key): values for key, values in self.q_table.items()},
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        self.q_table = {ast.literal_eval(key): values for key, values in payload["q_table"].items()}
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.alpha = payload.get("alpha", self.alpha)
        self.gamma = payload.get("gamma", self.gamma)
        self.epsilon_min = payload.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)
