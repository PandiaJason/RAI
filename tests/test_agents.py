import pytest
from rai.core.entity import Entity
from rai.core.agent import Agent

def test_agent_inventory():
    agent = Agent(id=1)
    e1 = Entity(1)
    
    agent.add_inventory(e1, 5.0)
    assert agent.inventory[e1] == 5.0
    
    agent.remove_inventory(e1, 2.0)
    assert agent.inventory[e1] == 3.0
    
    with pytest.raises(ValueError):
        agent.remove_inventory(e1, 5.0)
