import numpy as np
from rai.core.world import World
from rai.learning.env import RAIEnv

def calculate_gini(utilities: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a numpy array.
    """
    array = np.sort(utilities)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def measure_utility_inequality(world: World) -> float:
    """
    Calculates the Gini coefficient of utility across all agents.
    """
    env = RAIEnv(world)
    utilities = []
    
    for a_id in world.agents:
        utilities.append(env.calculate_utility(a_id))
        
    utils = np.array(utilities)
    if np.sum(utils) == 0:
        return 0.0
        
    return calculate_gini(utils)
