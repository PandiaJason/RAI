import csv
import random
from typing import Dict, Tuple
from rai.core.entity import Entity
from rai.core.relation import Relation
from rai.core.world import World
from rai.core.hypergraph import Hypergraph
from rai.core.agent import Agent
from rai.core.knowledge import Knowledge

class RealWorldParser:
    """
    Parses a real-world semantic dataset and completely strips the semantics,
    returning a purely abstract RAI World.
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
        
    def parse_csv(self, filepath: str, num_agents: int = 100) -> World:
        hypergraph = Hypergraph()
        
        # 1. Parse Relations and Entities
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                inputs_raw = row['Inputs'].split(';')
                outputs_raw = row['Outputs'].split(';')
                
                inputs = {}
                for item in inputs_raw:
                    if not item.strip(): continue
                    name, qty = item.split(':')
                    e_id = self._get_entity_id(name.strip())
                    inputs[Entity(e_id)] = float(qty)
                    
                outputs = {}
                for item in outputs_raw:
                    if not item.strip(): continue
                    name, qty = item.split(':')
                    e_id = self._get_entity_id(name.strip())
                    outputs[Entity(e_id)] = float(qty)
                    
                # Create relation
                rel = Relation(
                    id=self.next_relation_id,
                    inputs=inputs,
                    outputs=outputs,
                    knowledge_reqs={Knowledge(self.next_knowledge_id)} # Give each its own knowledge
                )
                self.next_relation_id += 1
                self.next_knowledge_id += 1
                
                hypergraph.add_relation(rel)
                
        # 2. Build World
        world = World()
        world.hypergraph = hypergraph
        
        # 3. Create Agents with random initial endowments
        all_entities = [Entity(i) for i in range(self.next_entity_id)]
        all_knowledge = [Knowledge(i) for i in range(self.next_knowledge_id)]
        
        for i in range(num_agents):
            agent = Agent(i)
            # Endow base resources
            for ent in all_entities:
                if random.random() < 0.5:
                    agent.inventory[ent] = random.uniform(5.0, 20.0)
            
            # Endow knowledge
            for k in all_knowledge:
                if random.random() < 0.3:
                    agent.knowledge.add(k)
                    
            # Preferences
            for ent in all_entities:
                agent.preferences[ent] = random.uniform(0.1, 1.0)
                
            world.add_agent(agent)
            
        return world
