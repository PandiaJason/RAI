import random
from typing import List
from rai.core.world import World
from rai.core.entity import Entity
from rai.core.agent import Agent
from rai.core.knowledge import Knowledge
from rai.core.relation import Relation

class WorldGenerator:
    """
    Procedurally generates synthetic relational worlds.
    """
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = random.randint(0, 999999)

    def generate(self, num_agents: int = 100, num_entities: int = 20, num_relations: int = 50, event_filepath: str = "rai_events.jsonl") -> World:
        world = World(event_filepath=event_filepath, seed=self.seed)
        
        # 1. Generate Entities
        entities = []
        for i in range(num_entities):
            e = Entity(id=i)
            entities.append(e)
            world.add_entity(e)
            
        # 2. Generate Knowledge States (optional, let's create a few)
        knowledges = []
        for i in range(num_relations // 2):
            k = Knowledge(id=i)
            knowledges.append(k)
            
        # 3. Generate Relations
        for i in range(num_relations):
            # Pick 1-3 inputs
            num_in = random.randint(1, 3)
            inputs = {}
            for _ in range(num_in):
                ent = random.choice(entities)
                qty = round(random.uniform(0.5, 3.0), 1)
                inputs[ent] = inputs.get(ent, 0) + qty
                
            # Pick 1-2 outputs
            num_out = random.randint(1, 2)
            outputs = {}
            for _ in range(num_out):
                ent = random.choice(entities)
                qty = round(random.uniform(0.5, 3.0), 1)
                outputs[ent] = outputs.get(ent, 0) + qty
                
            # Randomly require knowledge
            req_k = set()
            if knowledges and random.random() < 0.3:
                req_k.add(random.choice(knowledges))
                
            rel = Relation(
                id=i,
                inputs=inputs,
                outputs=outputs,
                knowledge_reqs=req_k
            )
            world.hypergraph.add_relation(rel)
            
        # 4. Generate Agents
        for i in range(num_agents):
            # Initial inventory: random subset of entities
            inv = {}
            for _ in range(random.randint(2, 5)):
                ent = random.choice(entities)
                inv[ent] = round(random.uniform(10.0, 50.0), 1)
                
            # Initial knowledge
            k_set = set()
            if knowledges and random.random() < 0.2:
                k_set.add(random.choice(knowledges))
                
            agent = Agent(id=i, initial_inventory=inv, initial_knowledge=k_set)
            
            # Random preferences (for utility calculations later)
            for e in entities:
                agent.preferences[e] = random.random()
                
            world.add_agent(agent)
            
        return world
