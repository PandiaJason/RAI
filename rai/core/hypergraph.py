from typing import Dict, List, Set, Optional
from .relation import Relation
from .entity import Entity

class Hypergraph:
    """
    The Dynamic Relational Hypergraph.
    Manages the set of all existing transformations (relations).
    """
    def __init__(self):
        self.relations: Dict[int, Relation] = {}
        
        # Fast lookups
        self._relations_by_input: Dict[Entity, Set[int]] = {}
        self._relations_by_output: Dict[Entity, Set[int]] = {}

    def add_relation(self, relation: Relation):
        self.relations[relation.id] = relation
        
        for ent in relation.inputs.keys():
            if ent not in self._relations_by_input:
                self._relations_by_input[ent] = set()
            self._relations_by_input[ent].add(relation.id)
            
        for ent in relation.outputs.keys():
            if ent not in self._relations_by_output:
                self._relations_by_output[ent] = set()
            self._relations_by_output[ent].add(relation.id)

    def get_relation(self, relation_id: int) -> Optional[Relation]:
        return self.relations.get(relation_id)

    def get_relations_consuming(self, entity: Entity) -> List[Relation]:
        rel_ids = self._relations_by_input.get(entity, set())
        return [self.relations[rid] for rid in rel_ids]

    def get_relations_producing(self, entity: Entity) -> List[Relation]:
        rel_ids = self._relations_by_output.get(entity, set())
        return [self.relations[rid] for rid in rel_ids]
        
    def get_all_relations(self) -> List[Relation]:
        return list(self.relations.values())
