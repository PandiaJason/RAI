import json
from collections import defaultdict
import math
from typing import List

def calculate_specialization_entropy(events_file: str) -> float:
    """
    Measures how specialized agents are by calculating the Shannon entropy
    of their transformation activities. Lower entropy = higher specialization.
    """
    agent_action_counts = defaultdict(lambda: defaultdict(int))
    
    with open(events_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            if data['event'] == 'TRANSFORM':
                agent = data['agent']
                rel = data['relation_id']
                agent_action_counts[agent][rel] += 1
                
    total_entropy = 0.0
    num_agents = 0
    
    for agent, counts in agent_action_counts.items():
        total_actions = sum(counts.values())
        if total_actions > 0:
            agent_ent = 0.0
            for rel, count in counts.items():
                p = count / total_actions
                agent_ent -= p * math.log(p)
            total_entropy += agent_ent
            num_agents += 1
            
    if num_agents == 0:
        return 0.0
        
    return total_entropy / num_agents
