import pytest
from rai.core.world import World
from rai.actions.explore import create_explore_action

def test_explore_action_logging():
    # Mostly we just verify that the explore action runs without error 
    # since discovery mechanics are hook-based.
    world = World(event_filepath="test_explore_events.jsonl")
    from rai.core.agent import Agent
    
    a1 = Agent(id=1)
    world.add_agent(a1)
    
    action = create_explore_action()
    world.step({1: action})
    assert world.tick == 1
    
    # Check log file
    with open("test_explore_events.jsonl", "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        assert "EXPLORE" in lines[0]
