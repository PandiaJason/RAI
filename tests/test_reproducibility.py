import pytest
import filecmp
import os
from rai.generation.world_generator import WorldGenerator
from rai.actions.transform import create_transform_action
import random

def run_simulation(seed: int, filepath: str):
    # Fixed sequence of actions for reproducibility test
    generator = WorldGenerator(seed=seed)
    world = generator.generate(num_agents=10, num_entities=5, num_relations=10, event_filepath=filepath)
    
    # Force some actions
    for _ in range(10):
        actions = {}
        for a_id in world.agents:
            # We use the world's PRNG implicitly via random.choice, 
            # so as long as we seeded it, it should be deterministic
            rel = random.choice(world.hypergraph.get_all_relations())
            actions[a_id] = create_transform_action(rel.id)
            
        world.step(actions)

def test_reproducibility():
    run_simulation(42, "run1.jsonl")
    run_simulation(42, "run2.jsonl")
    
    # Assert logs are identical
    assert filecmp.cmp("run1.jsonl", "run2.jsonl")
    
    # Clean up
    if os.path.exists("run1.jsonl"): os.remove("run1.jsonl")
    if os.path.exists("run2.jsonl"): os.remove("run2.jsonl")
