from typing import List, Dict, Any, Optional
import random

from .hypergraph import Hypergraph
from .agent import Agent
from .entity import Entity
from .events import EventLogger

class World:
    def __init__(self, event_filepath: str = "rai_events.jsonl", seed: int = None):
        self.hypergraph = Hypergraph()
        self.agents: Dict[int, Agent] = {}
        self.entities: Dict[int, Entity] = {}
        self.tick: int = 0
        self.logger = EventLogger(event_filepath)
        
        if seed is not None:
            random.seed(seed)

    def add_agent(self, agent: Agent):
        self.agents[agent.id] = agent

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity

    def step(self, agent_actions: Dict[int, Any]):
        """
        Advance the world state by one tick, resolving all actions.
        In Milestone 1-4, we process actions sequentially in a randomized order to handle conflicts.
        """
        self.tick += 1
        
        # Randomize order of agent actions to prevent positional bias
        agent_ids = list(agent_actions.keys())
        random.shuffle(agent_ids)

        for agent_id in agent_ids:
            action = agent_actions[agent_id]
            self._resolve_action(self.agents[agent_id], action)

    def _resolve_action(self, agent: Agent, action: Any):
        """
        Resolves a specific action. 
        For now, we will handle 'TRANSFORM', 'EXCHANGE', 'EXPLORE'.
        """
        if not action:
            return
            
        action_type = action.get('type')
        
        if action_type == 'TRANSFORM':
            # E.g., {'type': 'TRANSFORM', 'relation_id': 1}
            rel = self.hypergraph.get_relation(action.get('relation_id'))
            if rel and rel.can_execute(agent.inventory, agent.knowledge):
                # Consume inputs
                for ent, qty in rel.inputs.items():
                    agent.remove_inventory(ent, qty)
                # Produce outputs
                for ent, qty in rel.outputs.items():
                    agent.add_inventory(ent, qty)
                
                # Log event
                self.logger.log_event({
                    "time": self.tick,
                    "event": "TRANSFORM",
                    "agent": agent.id,
                    "relation_id": rel.id
                })

        elif action_type == 'EXCHANGE':
            # E.g., {'type': 'EXCHANGE', 'target_agent': 2, 'give_entity': 1, 'give_amount': 2.0, 'receive_entity': 2, 'receive_amount': 1.0}
            target = self.agents.get(action.get('target_agent'))
            give_ent = self.entities.get(action.get('give_entity'))
            recv_ent = self.entities.get(action.get('receive_entity'))
            
            give_amt = action.get('give_amount', 0.0)
            recv_amt = action.get('receive_amount', 0.0)
            
            # Simple atomic exchange logic (assumes target agent agrees automatically for this milestone's simplistic version)
            # In a real model, both must agree, or this is just a proposal.
            # We'll implement a forced swap for testing mechanics.
            if target and give_ent and recv_ent:
                if (agent.inventory.get(give_ent, 0.0) >= give_amt and 
                    target.inventory.get(recv_ent, 0.0) >= recv_amt):
                    
                    agent.remove_inventory(give_ent, give_amt)
                    target.add_inventory(give_ent, give_amt)
                    
                    target.remove_inventory(recv_ent, recv_amt)
                    agent.add_inventory(recv_ent, recv_amt)

                    self.logger.log_event({
                        "time": self.tick,
                        "event": "EXCHANGE",
                        "agent_1": agent.id,
                        "agent_2": target.id,
                        "give_entity": give_ent.id,
                        "give_amount": give_amt,
                        "receive_entity": recv_ent.id,
                        "receive_amount": recv_amt
                    })

        elif action_type == 'EXPLORE':
            # Agent tries to discover a new relation. (Handled via external logic usually, but we hook it here)
            # We log the attempt. The actual discovery success will be handled by the generation/explore logic.
            self.logger.log_event({
                "time": self.tick,
                "event": "EXPLORE",
                "agent": agent.id
            })
