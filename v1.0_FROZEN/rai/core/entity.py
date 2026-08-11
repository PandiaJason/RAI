from dataclasses import dataclass

@dataclass(frozen=True)
class Entity:
    """
    Represents an abstract entity/resource in the RAI universe.
    The entity has no semantic meaning (e.g., 'X1', 'X7').
    """
    id: int
    
    @property
    def name(self) -> str:
        return f"X{self.id}"
    
    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
