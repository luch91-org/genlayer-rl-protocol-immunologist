"""Tests for the tabular Q-learning agent. Everything here runs against a
scripted stand-in Env (or no Env at all) -- no MockEnv heuristic, no
network -- so these tests isolate agent.py's own logic: action selection,
the Bellman update, epsilon decay, serialization buckets, and Q-table
save/resume.
"""

from __future__ import annotations

import json

from agent.agent import (
    ACTIONS,
    QLearningAgent,
    action_name,
    serialize_state,
    strip_action,
)
from agent.train import run_episode

STATE_GREEN = {
    "treasury_balance": 1_000_000,
    "hedged_amount": 0,
    "is_paused": False,
    "signers": ["a", "b", "c"],
    "alert_level": "green",
    "threat_active": False,
    "round": 0,
    "total_score": 0.0,
    "last_reward": 0.0,
    "last_reason": "",
}
STATE_RED_PAUSED = {
    **STATE_GREEN,
    "alert_level": "red",
    "threat_active": True,
    "is_paused": True,
    "round": 3,
}


class ScriptedEnv:
    """Ignores whatever action it's given and returns a pre-scripted reward
    per step, transitioning through a fixed sequence of states."""

    def __init__(self, states: list[dict], rewards: list[float]):
        assert len(states) == len(rewards) + 1
        self.states = states
        self.rewards = rewards
        self._step_idx = 0

    def reset(self) -> dict:
        self._step_idx = 0
        return self.states[0]

    def step(self, action):
        reward = self.rewards[self._step_idx]
        self._step_idx += 1
        return reward, "scripted", self.states[self._step_idx]

    def is_episode_done(self) -> bool:
        return self._step_idx >= len(self.rewards)


# --- state serialization --------------------------------------------------------


def test_serialize_state_keys_on_alert_pause_hedge_and_damage():
    key = serialize_state(STATE_GREEN)
    assert key == ("green", False, "unhedged", "intact")


def test_serialize_state_hedge_buckets():
    partial = {**STATE_GREEN, "treasury_balance": 800_000, "hedged_amount": 200_000}
    heavy = {**STATE_GREEN, "treasury_balance": 500_000, "hedged_amount": 500_000}
    assert serialize_state(partial)[2] == "partial"
    assert serialize_state(heavy)[2] == "heavy"


def test_serialize_state_flags_damaged_treasury():
    damaged = {**STATE_GREEN, "treasury_balance": 700_000, "hedged_amount": 0}
    assert serialize_state(damaged)[3] == "damaged"


def test_distinct_alert_levels_produce_distinct_keys():
    yellow = {**STATE_GREEN, "alert_level": "yellow"}
    assert serialize_state(STATE_GREEN) != serialize_state(yellow)


# --- action helpers ---------------------------------------------------------------


def test_action_names_distinguish_hedge_sizes():
    names = [action_name(a) for a in ACTIONS]
    assert "hedge_small" in names
    assert "hedge_large" in names
    assert len(names) == len(set(names))


def test_strip_action_removes_internal_keys():
    hedge = next(a for a in ACTIONS if action_name(a) == "hedge_small")
    stripped = strip_action(hedge)
    assert "_name" not in stripped
    assert stripped["type"] == "hedge"
    assert stripped["amount"] == hedge["amount"]


# --- Q-learning mechanics ----------------------------------------------------------


def test_epsilon_decays_towards_the_floor():
    agent = QLearningAgent(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9)
    for _ in range(200):
        agent.decay_epsilon()
    assert agent.epsilon == 0.05


def test_update_moves_q_value_towards_a_positive_reward():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    key = serialize_state(STATE_GREEN)
    agent.update(STATE_GREEN, 0, reward=10.0, next_state=STATE_RED_PAUSED)
    assert agent.q_table[key][0] > 0.0


def test_bellman_update_matches_hand_computed_value():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    key = serialize_state(STATE_GREEN)
    agent.update(STATE_GREEN, 2, reward=10.0, next_state=STATE_RED_PAUSED)
    # td_target = 10 + 0.9 * 0; new_q = 0 + 0.5 * (10 - 0) = 5.0
    assert agent.q_table[key][2] == 5.0
    assert serialize_state(STATE_RED_PAUSED) in agent.q_table


def test_greedy_selection_picks_the_max_q_action():
    agent = QLearningAgent(epsilon_start=0.0, epsilon_min=0.0)
    key = serialize_state(STATE_GREEN)
    agent._ensure_state(key)
    agent.q_table[key][4] = 99.0
    idx, action = agent.select_action(STATE_GREEN)
    assert idx == 4
    assert action == ACTIONS[4]


def test_save_and_load_round_trips_q_table_and_hyperparameters(tmp_path):
    agent = QLearningAgent(alpha=0.2, gamma=0.8, epsilon_decay=0.95, epsilon_min=0.02)
    key = serialize_state(STATE_GREEN)
    agent._ensure_state(key)
    agent.q_table[key][1] = -1.5
    agent.epsilon = 0.42

    path = tmp_path / "q_table.json"
    agent.save(path)

    loaded = QLearningAgent()
    loaded.load(path)
    assert loaded.q_table == agent.q_table
    assert loaded.epsilon == 0.42
    assert loaded.alpha == 0.2
    assert loaded.gamma == 0.8


def test_saved_q_table_file_is_valid_json_with_string_keys(tmp_path):
    agent = QLearningAgent()
    agent._ensure_state(serialize_state(STATE_GREEN))
    path = tmp_path / "q_table.json"
    agent.save(path)
    raw = json.loads(path.read_text())
    assert all(isinstance(k, str) for k in raw["q_table"])


def test_mock_env_recognizes_every_agent_action():
    # Regression guard: strip_action removes _name, so MockEnv must
    # recover hedge_small/hedge_large from the amount. A gap here silently
    # scores an action 1.0 with reason 'unrecognized action'.
    from agent.env import MockEnv

    env = MockEnv(seed=0)
    for action in ACTIONS:
        env.reset()
        _reward, reason, _state = env.step(strip_action(action))
        assert reason != "unrecognized action", f"MockEnv can't score {action_name(action)}"


def test_run_episode_against_scripted_env_accumulates_rewards():
    env = ScriptedEnv([STATE_GREEN, STATE_RED_PAUSED, STATE_GREEN], [8.0, 2.0])
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon_start=1.0)
    episode_reward, last_reason = run_episode(env, agent)
    assert episode_reward == 10.0
    assert last_reason == "scripted"
    assert len(agent.q_table) == 2
