from dataclasses import dataclass, field
from typing import Dict
from .entity import Entity
from .knowledge import Knowledge

@dataclass(frozen=True)
class Relation:
    """
    Represents a transformation rule (hyperedge) in the relational universe.
    Example: 2X3 + X8 –[K4]→ 3X11
    """
    id: int
    inputs: Dict[Entity, float] = field(default_factory=dict)
    outputs: Dict[Entity, float] = field(default_factory=dict)
    knowledge_reqs: set[Knowledge] = field(default_factory=set)
    cost: float = 1.0 # Effort/Energy cost
    delay: int = 1 # Time ticks required for execution

    def __post_init__(self):
        # Allow passing integer IDs in dictionaries for convenience, 
        # but internally convert them to Entity objects to ensure type safety,
        # but since frozen=True we have to use object.__setattr__
        
        # We will assume inputs and outputs are proper Entity dicts for now
        pass

    def __repr__(self) -> str:
        in_str = " + ".join([f"{qty}{ent.name}" for ent, qty in self.inputs.items()])
        if not in_str: in_str = "0"
        
        out_str = " + ".join([f"{qty}{ent.name}" for ent, qty in self.outputs.items()])
        if not out_str: out_str = "0"
        
        k_str = f"-[{','.join([k.name for k in self.knowledge_reqs])}]->" if self.knowledge_reqs else "->"
        
        return f"R{self.id}: {in_str} {k_str} {out_str}"

    def can_execute(self, available_inventory: Dict[Entity, float], available_knowledge: set[Knowledge]) -> bool:
        """Checks if the provided inventory and knowledge satisfy the requirements."""
        # Check knowledge
        if not self.knowledge_reqs.issubset(available_knowledge):
            return False
            
        # Check inputs
        for ent, required_qty in self.inputs.items():
            if available_inventory.get(ent, 0.0) < required_qty:
                return False
                
        return True
