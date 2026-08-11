import csv
from typing import Dict
from rai.core.entity import Entity
from rai.core.relation import Relation
from rai.core.world import World
from rai.core.hypergraph import Hypergraph
from rai.core.knowledge import Knowledge

class FlightParser:
    """
    Parses OpenFlights dataset into abstract relations.
    """
    def __init__(self):
        self.entity_map: Dict[str, int] = {}
        self.next_entity_id = 0
        self.next_relation_id = 0
        self.next_knowledge_id = 0
        
    def _get_entity_id(self, name: str) -> int:
        if name not in self.entity_map:
            self.entity_map[name] = self.next_entity_id
            self.next_entity_id += 1
        return self.entity_map[name]
        
    def parse_csv(self, filepath: str) -> World:
        hypergraph = Hypergraph()
        
        with open(filepath, 'r') as f:
            # openflights format: Airline, AirlineID, Source, SourceID, Dest, DestID
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                source = parts[2]
                dest = parts[4]
                
                s_id = self._get_entity_id(source)
                d_id = self._get_entity_id(dest)
                
                rel = Relation(
                    id=self.next_relation_id,
                    inputs={Entity(s_id): 1.0},
                    outputs={Entity(d_id): 1.0},
                    knowledge_reqs={Knowledge(self.next_knowledge_id)}
                )
                hypergraph.add_relation(rel)
                self.next_relation_id += 1
                self.next_knowledge_id += 1
                
        world = World()
        world.hypergraph = hypergraph
        return world
