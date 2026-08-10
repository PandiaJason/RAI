from dataclasses import dataclass

@dataclass(frozen=True)
class Knowledge:
    """
    Represents an abstract knowledge state in the RAI universe.
    Knowledge enables specific transformations.
    """
    id: int
    
    @property
    def name(self) -> str:
        return f"K{self.id}"
    
    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
