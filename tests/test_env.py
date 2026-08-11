import pytest
import torch
from rai.core.world import World
from rai.core.agent import Agent
from rai.generation.world_generator import WorldGenerator
from rai.learning.env import RAIEnv

def test_env_observation_shape():
    generator = WorldGenerator(seed=42)
    world = generator.generate(num_agents=2, num_entities=5, num_relations=2)
    
    env = RAIEnv(world, max_entities=10, max_relations=5)
    
    obs, masks = env.get_observations()
    
    assert obs.shape == (2, 15) # 2 agents, 10 entities + 5 knowledge max
    assert masks.shape == (2, env.num_actions)

def test_env_step_utility():
    generator = WorldGenerator(seed=42)
    world = generator.generate(num_agents=2, num_entities=5, num_relations=2)
    env = RAIEnv(world, max_entities=10, max_relations=5)
    
    # 2 agents, ask both to WAIT (if they don't have enough resources for relation 0)
    # Actually just pass random actions
    actions = torch.tensor([0, 1])
    
    next_obs, next_masks, rewards = env.step(actions)
    
    assert rewards.shape == (2,)
