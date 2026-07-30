#!/usr/bin/env python
"""
Test SmartAgent with KGDecisionHelper integration
"""

import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_bktree_state_encoder():
    """Test BKTreeStateEncoder [NEW BKTree State Encoding]"""
    print("\n" + "=" * 60)
    print("Test 1: BKTreeStateEncoder - [NEW] BKTree State Encoding")
    print("=" * 60)

    from src.env.agents.SmartAgent import BKTreeStateEncoder

    state_node_dict = {
        (4, 4): {"id": 0, "score": 10.0},
        (3, 4): {"id": 1, "score": 5.0},
        (4, 3): {"id": 2, "score": -5.0},
        (2, 4): {"id": 3, "score": -10.0},
    }

    encoder = BKTreeStateEncoder(state_node_dict=state_node_dict)

    state_id = encoder.encode(4, 4)
    print(f"  encode(4, 4) -> state_id={state_id} (expected: 0)")
    assert state_id == 0, f"Expected 0, got {state_id}"

    state_id = encoder.encode(3, 4)
    print(f"  encode(3, 4) -> state_id={state_id} (expected: 1)")
    assert state_id == 1, f"Expected 1, got {state_id}"

    state_id = encoder.encode(100, 100)
    print(f"  encode(100, 100) -> state_id={state_id} (expected: None)")
    assert state_id is None, f"Expected None, got {state_id}"

    print("  [PASS] BKTreeStateEncoder tests passed!")


def test_smart_agent_init():
    """Test SmartAgent initialization"""
    print("\n" + "=" * 60)
    print("Test 2: SmartAgent Initialization")
    print("=" * 60)

    try:
        from src.env.agents.SmartAgent import SmartAgent

        agent = SmartAgent(
            kg_path="cache/knowledge_graph/kg_simple.pkl",
            strategy="roulette",
            use_kg_helper=True,
        )

        print(f"  Strategy: {agent.strategy}")
        print(f"  Use KG Helper: {agent.use_kg_helper}")
        print(f"  State Encoder: {type(agent.state_encoder).__name__}")

        if hasattr(agent.state_encoder, "state_node_dict"):
            num_states = len(agent.state_encoder.state_node_dict)
            print(f"  Loaded states: {num_states}")

        print("  [PASS] SmartAgent initialization passed!")
        return agent

    except Exception as e:
        print(f"  [FAIL] SmartAgent initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_state_encoding_with_real_data(agent):
    """Test state encoding with real data"""
    print("\n" + "=" * 60)
    print("Test 3: State Encoding with Real Data")
    print("=" * 60)

    if agent is None:
        print("  [SKIP] Agent not initialized")
        return

    test_cases = [
        (4, 4, "typical opening"),
        (3, 3, "each lost 1 unit"),
        (2, 2, "each lost 2 units"),
        (1, 1, "end game"),
        (0, 0, "extreme case"),
    ]

    for my_count, enemy_count, desc in test_cases:
        state_id = agent.state_encoder.encode(my_count, enemy_count)
        if state_id is not None:
            print(f"  ({my_count}, {enemy_count}) -> state_id={state_id} [{desc}]")
        else:
            print(f"  ({my_count}, {enemy_count}) -> NOT FOUND [{desc}]")


def test_kg_helper_integration(agent):
    """Test KGDecisionHelper integration"""
    print("\n" + "=" * 60)
    print("Test 4: KGDecisionHelper Integration")
    print("=" * 60)

    if agent is None or agent.kg_helper is None:
        print("  [SKIP] KGHelper not available")
        return

    test_states = [0, 109, 544, 396]

    for state_id in test_states:
        action, info = agent.kg_helper.select_action(state_id, k=5, strategy="roulette")
        if action:
            quality = info.get("quality_score", 0)
            print(f"  State {state_id}: action={action}, quality={quality:.2f}")
        else:
            print(f"  State {state_id}: No action recommended")


def test_action_parsing(agent):
    """Test action string parsing"""
    print("\n" + "=" * 60)
    print("Test 5: Action String Parsing")
    print("=" * 60)

    if agent is None:
        print("  [SKIP] Agent not initialized")
        return

    from src.env.agents.SmartAgent import ACTION_NAMES, CLUSTERS

    test_actions = ["4d", "2b", "0a", "4k"]

    for action_str in test_actions:
        try:
            cluster_idx = int(action_str[0])
            action_letter = action_str[1]

            cluster_method = CLUSTERS[cluster_idx]
            action_method = ACTION_NAMES.get(action_letter, "unknown")

            print(
                f"  '{action_str}' -> cluster={cluster_method}, action={action_method}"
            )
        except (ValueError, IndexError) as e:
            print(f"  '{action_str}' -> ERROR: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("SmartAgent Test Suite")
    print("Testing BKTree State Encoder and KGDecisionHelper Integration")
    print("=" * 60)

    test_bktree_state_encoder()
    agent = test_smart_agent_init()
    test_state_encoding_with_real_data(agent)
    test_kg_helper_integration(agent)
    test_action_parsing(agent)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
