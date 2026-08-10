from typing import Dict, List, Set, Any
from .entity import Entity
from .knowledge import Knowledge

class Agent:
    def __init__(self, id: int, initial_inventory: Dict[Entity, float] = None, initial_knowledge: Set[Knowledge] = None):
        self.id = id
        self.inventory: Dict[Entity, float] = initial_inventory or {}
        self.knowledge: Set[Knowledge] = initial_knowledge or set()
        
        # Local state metrics
        self.utility: float = 0.0
        
        # In a real implementation this would hold the neural network or heuristic policy
        self.preferences: Dict[Entity, float] = {}

    def add_inventory(self, entity: Entity, amount: float):
        if entity not in self.inventory:
            self.inventory[entity] = 0.0
        self.inventory[entity] += amount

    def remove_inventory(self, entity: Entity, amount: float):
        if self.inventory.get(entity, 0.0) < amount:
            raise ValueError(f"Agent {self.id} does not have enough {entity} to remove {amount}.")
        self.inventory[entity] -= amount
        
    def add_knowledge(self, knowledge: Knowledge):
        self.knowledge.add(knowledge)
        
    def get_observation(self, world: Any) -> Dict[str, Any]:
        """
        Partial observation of the world. 
        For now, just local state.
        """
        return {
            "inventory": {e.id: qty for e, qty in self.inventory.items()},
            "knowledge": [k.id for k in self.knowledge]
        }
