import pytest
from rai.core.entity import Entity
from rai.core.agent import Agent
from rai.core.world import World
from rai.actions.exchange import create_exchange_action

def test_exchange_action():
    world = World(event_filepath="test_exchange_events.jsonl")
    
    e1 = Entity(1)
    e2 = Entity(2)
    world.add_entity(e1)
    world.add_entity(e2)
    
    a1 = Agent(id=1, initial_inventory={e1: 10.0, e2: 0.0})
    a2 = Agent(id=2, initial_inventory={e1: 0.0, e2: 10.0})
    
    world.add_agent(a1)
    world.add_agent(a2)
    
    # a1 wants to give 2.0 e1 to a2 in exchange for 3.0 e2
    action = create_exchange_action(
        target_agent_id=2,
        give_entity_id=1,
        give_amount=2.0,
        receive_entity_id=2,
        receive_amount=3.0
    )
    
    world.step({1: action})
    
    assert world.agents[1].inventory[e1] == 8.0
    assert world.agents[1].inventory[e2] == 3.0
    assert world.agents[2].inventory[e1] == 2.0
    assert world.agents[2].inventory[e2] == 7.0
