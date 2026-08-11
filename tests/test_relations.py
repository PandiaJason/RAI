import pytest
from rai.core.entity import Entity
from rai.core.knowledge import Knowledge
from rai.core.relation import Relation

def test_relation_can_execute():
    e1 = Entity(1)
    e2 = Entity(2)
    k1 = Knowledge(1)
    
    rel = Relation(
        id=1,
        inputs={e1: 2.0},
        outputs={e2: 1.0},
        knowledge_reqs={k1}
    )
    
    # Missing knowledge and inventory
    assert not rel.can_execute({}, set())
    
    # Has inventory, missing knowledge
    assert not rel.can_execute({e1: 2.0}, set())
    
    # Has knowledge, missing inventory
    assert not rel.can_execute({}, {k1})
    
    # Has both
    assert rel.can_execute({e1: 2.5}, {k1})
