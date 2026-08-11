import pytest
from rai.core.entity import Entity
from rai.core.agent import Agent
from rai.core.world import World

def test_world_initialization():
    world = World()
    assert world.tick == 0
    assert len(world.agents) == 0
    assert len(world.entities) == 0

def test_world_step():
    world = World(event_filepath="test_events.jsonl")
    world.step({})
    assert world.tick == 1
    world.step({})
    assert world.tick == 2
