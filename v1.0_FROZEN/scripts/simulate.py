import argparse
import random
from rai.generation.world_generator import WorldGenerator
from rai.actions.transform import create_transform_action
from rai.actions.exchange import create_exchange_action
from rai.actions.explore import create_explore_action

def run_simulation(agents: int, entities: int, relations: int, steps: int, seed: int, log_file: str):
    print(f"Initializing RAI Simulation (Smoke Test)")
    print(f"Agents: {agents} | Entities: {entities} | Relations: {relations} | Seed: {seed}")
    
    generator = WorldGenerator(seed=seed)
    world = generator.generate(num_agents=agents, num_entities=entities, num_relations=relations, event_filepath=log_file)
    
    all_relations = world.hypergraph.get_all_relations()
    agent_ids = list(world.agents.keys())
    
    print(f"Starting simulation for {steps} steps...")
    
    for step in range(steps):
        actions = {}
        for a_id in agent_ids:
            # Without RL, we just have agents perform random valid actions for testing
            action_choice = random.choice(["TRANSFORM", "EXCHANGE", "EXPLORE", "WAIT"])
            
            if action_choice == "TRANSFORM" and all_relations:
                rel = random.choice(all_relations)
                actions[a_id] = create_transform_action(rel.id)
                
            elif action_choice == "EXCHANGE" and len(agent_ids) > 1:
                target_id = random.choice([i for i in agent_ids if i != a_id])
                # Randomize entities to exchange
                give_ent = random.randint(0, entities - 1)
                recv_ent = random.randint(0, entities - 1)
                actions[a_id] = create_exchange_action(target_id, give_ent, 1.0, recv_ent, 1.0)
                
            elif action_choice == "EXPLORE":
                actions[a_id] = create_explore_action()
                
        world.step(actions)
        
        if (step + 1) % (max(1, steps // 10)) == 0:
            print(f"Step {step + 1}/{steps} completed.")
            
    print(f"Simulation finished. Event log saved to: {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a basic RAI smoke test.")
    parser.add_argument("--agents", type=int, default=100, help="Number of agents")
    parser.add_argument("--entities", type=int, default=20, help="Number of entities")
    parser.add_argument("--relations", type=int, default=50, help="Number of initial relations")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log", type=str, default="results/smoke_test_events.jsonl", help="Log output path")
    
    args = parser.parse_args()
    
    import os
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    
    run_simulation(args.agents, args.entities, args.relations, args.steps, args.seed, args.log)
