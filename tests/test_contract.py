"""Tests for the deterministic parts of ProtocolImmunologist.

These exercise the REAL contract source via contracts/logic.py's stubbed
loader -- no GenVM runtime involved. We do NOT assert on exact LLM reward
scores anywhere: the LLM judge is non-deterministic by design, so instead
we test (a) the deterministic state machine that runs identically on every
validator, and (b) the reward-parsing/bounds logic.
"""

from __future__ import annotations

import json

import pytest

from contracts.logic import (
    DRAIN_DIVISOR,
    INITIAL_SIGNERS,
    INITIAL_TREASURY,
    THREAT_SCHEDULE,
    ProtocolImmunologist,
    alert_for_round,
    apply_action,
    build_reward_prompt,
    initial_state,
    normalize_reward_for_consensus,
    parse_reward_output,
    score_to_x100,
)


def state_with(**overrides):
    state = initial_state()
    state.update(overrides)
    return state


# --- action transitions ------------------------------------------------------


def test_pause_sets_is_paused():
    new_state, loss, applied = apply_action(initial_state(), {"type": "pause"}, 1)
    assert applied is True
    assert new_state["is_paused"] is True
    assert loss == 0


def test_unpause_clears_is_paused():
    new_state, _loss, applied = apply_action(state_with(is_paused=True), {"type": "unpause"}, 1)
    assert applied is True
    assert new_state["is_paused"] is False


def test_rotate_signer_drops_oldest_and_appends_new():
    new_state, _loss, applied = apply_action(
        initial_state(), {"type": "rotate_signer", "new_signer": "signer_fresh"}, 1
    )
    assert applied is True
    assert new_state["signers"] == [INITIAL_SIGNERS[1], INITIAL_SIGNERS[2], "signer_fresh"]
    assert len(new_state["signers"]) == 3


def test_rotate_signer_without_name_generates_one_from_the_round():
    new_state, _loss, _applied = apply_action(initial_state(), {"type": "rotate_signer"}, 7)
    assert new_state["signers"][-1] == "signer_r7"


def test_hedge_moves_funds_from_treasury_to_hedged():
    new_state, _loss, applied = apply_action(
        initial_state(), {"type": "hedge", "amount": 250_000}, 1
    )
    assert applied is True
    assert new_state["treasury_balance"] == INITIAL_TREASURY - 250_000
    assert new_state["hedged_amount"] == 250_000


def test_hedge_is_capped_at_the_treasury_balance():
    new_state, _loss, _applied = apply_action(
        initial_state(), {"type": "hedge", "amount": INITIAL_TREASURY * 5}, 1
    )
    assert new_state["treasury_balance"] == 0
    assert new_state["hedged_amount"] == INITIAL_TREASURY


def test_hedge_rejects_negative_amounts():
    new_state, _loss, _applied = apply_action(initial_state(), {"type": "hedge", "amount": -500}, 1)
    assert new_state["treasury_balance"] == INITIAL_TREASURY
    assert new_state["hedged_amount"] == 0


def test_do_nothing_is_a_no_op_on_funds_and_flags():
    new_state, _loss, applied = apply_action(initial_state(), {"type": "do_nothing"}, 1)
    assert applied is False
    assert new_state["treasury_balance"] == INITIAL_TREASURY
    assert new_state["is_paused"] is False
    assert new_state["signers"] == list(INITIAL_SIGNERS)


def test_apply_action_does_not_mutate_its_input():
    state = initial_state()
    snapshot = json.loads(json.dumps(state))
    apply_action(state, {"type": "hedge", "amount": 100_000}, 1)
    assert state == snapshot


# --- threat consequence -------------------------------------------------------


def test_red_alert_drains_unhedged_treasury_when_not_paused():
    new_state, loss, _applied = apply_action(
        state_with(alert_level="red"), {"type": "do_nothing"}, 1
    )
    assert loss == INITIAL_TREASURY // DRAIN_DIVISOR
    assert new_state["treasury_balance"] == INITIAL_TREASURY - loss


def test_red_alert_does_not_drain_when_paused_this_round():
    # Pausing IN the red round protects immediately.
    new_state, loss, _applied = apply_action(state_with(alert_level="red"), {"type": "pause"}, 1)
    assert loss == 0
    assert new_state["treasury_balance"] == INITIAL_TREASURY


def test_red_alert_spares_hedged_funds():
    hedged_state = state_with(alert_level="red", treasury_balance=400_000, hedged_amount=600_000)
    new_state, loss, _applied = apply_action(hedged_state, {"type": "do_nothing"}, 1)
    assert loss == 400_000 // DRAIN_DIVISOR
    assert new_state["hedged_amount"] == 600_000


def test_green_and_yellow_alerts_never_drain():
    for level in ("green", "yellow"):
        _new_state, loss, _applied = apply_action(
            state_with(alert_level=level), {"type": "do_nothing"}, 1
        )
        assert loss == 0


# --- alert schedule -----------------------------------------------------------


def test_alert_advances_along_the_deterministic_schedule():
    new_state, _loss, _applied = apply_action(initial_state(), {"type": "do_nothing"}, 3)
    assert new_state["alert_level"] == THREAT_SCHEDULE[3 % len(THREAT_SCHEDULE)]


def test_alert_for_round_wraps_around_the_schedule():
    assert alert_for_round(0) == THREAT_SCHEDULE[0]
    assert alert_for_round(len(THREAT_SCHEDULE)) == THREAT_SCHEDULE[0]
    assert alert_for_round(len(THREAT_SCHEDULE) + 2) == THREAT_SCHEDULE[2]


def test_schedule_contains_every_alert_level():
    # The judge rubric needs the agent to experience all three regimes.
    assert set(THREAT_SCHEDULE) == {"green", "yellow", "red"}


# --- reward parsing / bounds ----------------------------------------------------


def test_parse_reward_output_accepts_a_dict():
    score, reason = parse_reward_output({"score": 7, "reason": "solid call"})
    assert score == 7.0
    assert reason == "solid call"


def test_parse_reward_output_accepts_a_json_string():
    score, reason = parse_reward_output(json.dumps({"score": 4.5, "reason": "meh"}))
    assert score == 4.5
    assert reason == "meh"


def test_parse_reward_output_clamps_out_of_range_scores():
    assert parse_reward_output({"score": 15})[0] == 10.0
    assert parse_reward_output({"score": -3})[0] == 0.0


def test_parse_reward_output_rejects_missing_score():
    with pytest.raises(KeyError):
        parse_reward_output({"reason": "no score field"})


def test_score_to_x100_scales_and_rounds():
    assert score_to_x100(7.5) == 750
    assert score_to_x100(0.0) == 0
    assert score_to_x100(10.0) == 1000
    assert score_to_x100(6.666) == 667  # rounds, never truncates


def test_normalize_reward_for_consensus_is_stable_json():
    normalized = normalize_reward_for_consensus(6.0, "ok")
    assert normalized == normalize_reward_for_consensus(6.0, "ok")
    assert json.loads(normalized) == {"score": 6.0, "reason": "ok"}


def test_build_reward_prompt_includes_all_snapshotted_fields():
    prompt = build_reward_prompt(
        '{"before": 1}', '{"after": 2}', '{"type": "pause"}', "red", 5000, 4
    )
    assert '{"before": 1}' in prompt
    assert '{"after": 2}' in prompt
    assert '{"type": "pause"}' in prompt
    assert "red" in prompt
    assert "5000" in prompt
    assert "Round: 4" in prompt


# --- contract wiring (real source, stubbed genlayer runtime) -------------------


def test_contract_source_is_self_contained():
    # deploy_contract(code=...) sends exactly one file on-chain, so the
    # contract must never import sibling modules.
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parent.parent / "contracts" / "protocol_immunologist.py"
    ).read_text(encoding="utf-8")
    assert "from contracts" not in source
    assert "import contracts" not in source
    assert "import agent" not in source


def test_contract_initializes_with_expected_state():
    contract = ProtocolImmunologist()
    state = contract.get_state()
    assert state["treasury_balance"] == INITIAL_TREASURY
    assert state["hedged_amount"] == 0
    assert state["is_paused"] is False
    assert state["signers"] == list(INITIAL_SIGNERS)
    assert state["alert_level"] == "green"
    assert state["threat_active"] is False
    assert state["round"] == 0
    assert state["total_score_x100"] == 0
    assert state["last_reward_x100"] == 0
    assert contract.get_score() == 0


def test_contract_take_action_cannot_run_off_chain():
    # The nondet/eq_principle stubs must raise, so nothing off-chain can
    # accidentally "succeed" at calling the LLM judge.
    contract = ProtocolImmunologist()
    with pytest.raises(NotImplementedError):
        contract.take_action({"type": "do_nothing"})
