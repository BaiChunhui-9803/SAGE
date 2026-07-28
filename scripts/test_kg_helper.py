#!/usr/bin/env python
"""
Test KGDecisionHelper class
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.decision.kg_decision_helper import KGDecisionHelper


def test_basic_selection():
    print("\n=== Test 1: Basic Action Selection ===")
    helper = KGDecisionHelper("cache/knowledge_graph/kg_simple.pkl")

    test_states = [109, 544, 396, 0, 100]

    for state in test_states:
        action, info = helper.select_action(state, k=5, strategy="roulette")
        if action:
            print(
                f"State {state}: action={action}, quality={info.get('quality_score', 0):.2f}"
            )
        else:
            print(f"State {state}: No valid action found")


def test_top_k_actions():
    print("\n=== Test 2: Get Top-K Actions ===")
    helper = KGDecisionHelper("cache/knowledge_graph/kg_simple.pkl")

    state = 109
    top_actions = helper.get_top_k_actions(state, k=5)

    print(f"Top actions for state {state}:")
    for action, info in top_actions:
        print(
            f"  {action}: quality={info['quality_score']:.2f}, visits={info['visits']}"
        )


def test_predict_trajectory():
    print("\n=== Test 3: Predict Trajectory ===")
    helper = KGDecisionHelper("cache/knowledge_graph/kg_simple.pkl")

    state = 109
    top_actions = helper.get_top_k_actions(state, k=1)
    if top_actions:
        action = top_actions[0][0]
        trajectory = helper.predict_trajectory(state, action, steps=5)
        print(f"Trajectory from state {state}, action {action}:")
        for step_info in trajectory:
            print(
                f"  Step {step_info['step']}: state={step_info['state']}, "
                f"action={step_info['action']}, quality={step_info['quality']:.2f}"
            )


def test_win_probability():
    print("\n=== Test 4: Win Probability ===")
    helper = KGDecisionHelper("cache/knowledge_graph/kg_simple.pkl")

    state = 109
    top_actions = helper.get_top_k_actions(state, k=3)

    print(f"Win probability for state {state}:")
    for action, info in top_actions:
        win_prob = helper.get_winning_probability(state, action)
        future_reward = helper.get_expected_future_reward(state, action)
        print(f"  {action}: win_rate={win_prob:.2%}, future_reward={future_reward:.2f}")


def test_predict_next_states():
    print("\n=== Test 5: Predict Next States ===")
    helper = KGDecisionHelper("cache/knowledge_graph/kg_simple.pkl")

    state = 109
    top_actions = helper.get_top_k_actions(state, k=1)
    if top_actions:
        action = top_actions[0][0]
        next_states = helper.predict_next_states(state, action)
        print(f"Next states from state {state}, action {action}:")
        for next_state, prob in sorted(next_states.items(), key=lambda x: -x[1])[:5]:
            print(f"  -> State {next_state}: {prob:.2%}")


if __name__ == "__main__":
    test_basic_selection()
    test_top_k_actions()
    test_predict_trajectory()
    test_win_probability()
    test_predict_next_states()
    print("\n=== All tests completed ===")
