import numpy as np
from typing import Dict, Any, List, Tuple
from rai.core.world import World
from rai.actions.transform import create_transform_action
from rai.actions.explore import create_explore_action
from rai.actions.exchange import create_exchange_action
import torch
import math

class RAIEnv:
    """
    A wrapper around World to interface with the PPO agent.
    Maps discrete action indices to actual RAI actions.
    Calculates agent-local utility and rewards.
    """
    def __init__(self, world: World, max_entities: int = 100, max_relations: int = 500):
        self.world = world
        self.max_entities = max_entities
        self.max_relations = max_relations
        
        # Action space: 
        # 0 to max_relations - 1: TRANSFORM relations
        # max_relations: EXPLORE
        # max_relations + 1 to max_relations + 1 + (max_entities): Random EXCHANGE proposals (simplified)
        self.num_actions = self.max_relations + 1 + self.max_entities

    def get_obs_dim(self) -> int:
        # Obs = Inventory (max_entities) + Knowledge flags (max_relations)
        return self.max_entities + self.max_relations

    def get_observations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a batched tensor of observations and action masks for all agents.
        """
        agent_ids = sorted(list(self.world.agents.keys()))
        batch_size = len(agent_ids)
        
        obs_tensor = torch.zeros((batch_size, self.get_obs_dim()))
        mask_tensor = torch.zeros((batch_size, self.num_actions))
        
        for idx, a_id in enumerate(agent_ids):
            agent = self.world.agents[a_id]
            
            # Fill inventory part of obs
            for ent, qty in agent.inventory.items():
                if ent.id < self.max_entities:
                    obs_tensor[idx, ent.id] = qty
                    
            # Fill knowledge part
            for k in agent.knowledge:
                if k.id < self.max_relations:
                    obs_tensor[idx, self.max_entities + k.id] = 1.0
                    
            # Build action mask
            # Can always explore
            mask_tensor[idx, self.max_relations] = 1.0
            
            # Can transform if knowledge + inventory allows
            for rel in self.world.hypergraph.get_all_relations():
                if rel.id < self.max_relations:
                    if rel.can_execute(agent.inventory, agent.knowledge):
                        mask_tensor[idx, rel.id] = 1.0
                        
            # Exchange actions (simplified mask, always true for now)
            mask_tensor[idx, self.max_relations+1:] = 1.0
            
        return obs_tensor, mask_tensor

    def calculate_utility(self, agent_id: int) -> float:
        """U_i(t) = sum(w_ij * log(1 + x_ij))"""
        agent = self.world.agents[agent_id]
        u = 0.0
        for ent, qty in agent.inventory.items():
            pref = agent.preferences.get(ent, 0.1)
            u += pref * math.log(1.0 + max(0, qty))
        return u

    def step(self, actions_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes batched actions from PPO, advances world, calculates rewards.
        """
        agent_ids = sorted(list(self.world.agents.keys()))
        actions_dict = {}
        
        # Calculate utility before
        u_before = {a_id: self.calculate_utility(a_id) for a_id in agent_ids}
        
        # Translate network discrete actions to RAI actions
        for idx, a_id in enumerate(agent_ids):
            act_idx = actions_tensor[idx].item()
            
            if act_idx < self.max_relations:
                actions_dict[a_id] = create_transform_action(act_idx)
            elif act_idx == self.max_relations:
                actions_dict[a_id] = create_explore_action()
            else:
                # Exchange mapping (very simplified for milestone)
                target_ent = (act_idx - self.max_relations - 1) % self.max_entities
                actions_dict[a_id] = create_exchange_action(
                    target_agent_id=(a_id + 1) % len(agent_ids), # just random neighbor
                    give_entity_id=0, # Simplified
                    give_amount=1.0,
                    receive_entity_id=target_ent,
                    receive_amount=1.0
                )
                
        # Step world
        self.world.step(actions_dict)
        
        # Calculate utility after
        rewards = torch.zeros(len(agent_ids))
        for idx, a_id in enumerate(agent_ids):
            u_after = self.calculate_utility(a_id)
            rewards[idx] = u_after - u_before[a_id]
            
        # Get new obs
        next_obs, next_masks = self.get_observations()
        
        return next_obs, next_masks, rewards
