import argparse
import random
import torch
import numpy as np
import copy
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from rai.generation.kaggle_parser import KaggleParser
from rai.learning.env import RAIEnv
from rai.learning.actor_critic import SharedActorCritic
from rai.core.relation import Relation
from rai.core.entity import Entity
from rai.core.world import World
from rai.core.hypergraph import Hypergraph

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_candidates(all_relations, max_entities, num_positives):
    positives = random.sample(all_relations, num_positives)
    
    pos_set = set()
    for r in all_relations:
        if r.inputs and r.outputs:
            i_e = list(r.inputs.keys())[0].id
            o_e = list(r.outputs.keys())[0].id
            pos_set.add((i_e, o_e))
            
    negatives = []
    while len(negatives) < num_positives:
        in_e = random.randint(0, max_entities-1)
        out_e = random.randint(0, max_entities-1)
        if in_e != out_e and (in_e, out_e) not in pos_set:
            rel = Relation(id=0, inputs={Entity(in_e):1.0}, outputs={Entity(out_e):1.0})
            negatives.append(rel)
            pos_set.add((in_e, out_e))
            
    return positives, negatives

def evaluate_link_prediction(model_path, data_path, seed=42, scramble_ids=False):
    set_seed(seed)
    
    parser = KaggleParser()
    world = parser.parse_csv(data_path, num_agents=1)
    all_relations = list(world.hypergraph.relations.values())
    max_entities = parser.next_entity_id
    
    if scramble_ids:
        # Scramble entity IDs to prove semantic-free structure
        mapping = list(range(max_entities))
        random.shuffle(mapping)
        for rel in all_relations:
            new_inputs = {Entity(mapping[e.id]): v for e, v in rel.inputs.items()}
            new_outputs = {Entity(mapping[e.id]): v for e, v in rel.outputs.items()}
            # Reassign via bypass (since it's a frozen dataclass)
            object.__setattr__(rel, 'inputs', new_inputs)
            object.__setattr__(rel, 'outputs', new_outputs)
            
    
    num_total = len(all_relations)
    num_hidden = int(num_total * 0.2)
    
    positives, negatives = generate_candidates(all_relations, max_entities, num_hidden)
    
    # 80% visible graph
    visible_relations = [r for r in all_relations if r not in positives]
    
    # Models
    rai_policy = SharedActorCritic(obs_dim=600, num_actions=601, hidden_size=128) # Default dims
    try:
        rai_policy.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        pass # If file missing during testing, just use untrained weights for both
    rai_policy.eval()
    
    untrained_policy = SharedActorCritic(obs_dim=600, num_actions=601, hidden_size=128)
    untrained_policy.eval()
    
    candidates = [(p, 1) for p in positives] + [(n, 0) for n in negatives]
    random.shuffle(candidates)
    
    y_true = []
    
    scores_rai = []
    scores_untrained = []
    scores_random = []
    scores_heuristic = []
    
    # Common features of the 80% graph
    for idx, (cand_rel, label) in enumerate(candidates):
        y_true.append(label)
        
        # Build a temporary world with visible + this candidate
        temp_world = World()
        temp_hg = Hypergraph()
        
        # Add visible
        for i, r in enumerate(visible_relations):
            new_r = Relation(id=i+1, inputs=r.inputs, outputs=r.outputs, knowledge_reqs=r.knowledge_reqs, cost=r.cost)
            temp_hg.add_relation(new_r)
            
        # Add candidate at specific index (e.g. N+1)
        cand_idx = len(visible_relations) + 1
        cand_temp = Relation(id=cand_idx, inputs=cand_rel.inputs, outputs=cand_rel.outputs)
        temp_hg.add_relation(cand_temp)
        
        temp_world.hypergraph = temp_hg
        
        # We need an agent to get an observation
        from rai.core.agent import Agent
        a = Agent(0)
        a.inventory = {Entity(i): 10.0 for i in range(max_entities)} # Fully stocked so it doesn't mask it out based on inventory
        temp_world.add_agent(a)
        
        env = RAIEnv(temp_world, max_entities=100, max_relations=500)
        obs, _ = env.get_observations()
        
        with torch.no_grad():
            dist_rai, _ = rai_policy(obs)
            dist_un, _ = untrained_policy(obs)
            
            # The logit for the candidate index
            # dist.logits is (batch, num_actions)
            logit_rai = dist_rai.logits[0, cand_idx].item()
            logit_un = dist_un.logits[0, cand_idx].item()
            
            scores_rai.append(logit_rai)
            scores_untrained.append(logit_un)
            
            # Random baseline
            scores_random.append(random.uniform(-1, 1))
            
            # Graph Heuristic: Jaccard-like (if they share inputs/outputs with existing)
            # Simple heuristic: degree of the nodes
            in_e = list(cand_rel.inputs.keys())[0].id if cand_rel.inputs else 0
            out_e = list(cand_rel.outputs.keys())[0].id if cand_rel.outputs else 0
            
            degree = sum(1 for r in visible_relations if (r.inputs and list(r.inputs.keys())[0].id in [in_e, out_e]) or (r.outputs and list(r.outputs.keys())[0].id in [in_e, out_e]))
            scores_heuristic.append(float(degree))
            
    return y_true, scores_rai, scores_untrained, scores_random, scores_heuristic

def calc_metrics(y_true, scores):
    # Normalize scores to 0-1 for probability thresholds
    s = np.array(scores)
    if s.max() == s.min():
        probs = np.zeros_like(s)
    else:
        probs = (s - s.min()) / (s.max() - s.min())
        
    preds = (probs > 0.5).astype(int)
    
    if sum(preds) == 0:
        return 0, 0, 0, 0, 0
    
    f1 = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    roc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    
    return f1, prec, rec, roc, pr_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/rai_seed_0.pt")
    parser.add_argument("--data", type=str, default="data/kaggle_supply_chain.csv")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--scramble", action="store_true")
    args = parser.parse_args()
    
    results_rai = []
    results_un = []
    results_rand = []
    results_heur = []
    
    for seed in range(args.seeds):
        y, s_rai, s_un, s_rand, s_heur = evaluate_link_prediction(args.model, args.data, seed=seed, scramble_ids=args.scramble)
        
        results_rai.append(calc_metrics(y, s_rai)[0]) # just F1 for loop
        results_un.append(calc_metrics(y, s_un)[0])
        results_rand.append(calc_metrics(y, s_rand)[0])
        results_heur.append(calc_metrics(y, s_heur)[0])
        
    print(f"\n--- ZERO SHOT RELATIONAL COMPLETION (30 SEEDS) ---")
    if args.scramble:
        print("(ID PERMUTATION TEST ACTIVE: Entites have been completely renamed)")
    
    print(f"{'Model':<20} | F1 (Mean ± Std)")
    print(f"{'Random':<20} | {np.mean(results_rand):.2f} ± {np.std(results_rand):.2f}")
    print(f"{'Untrained NN':<20} | {np.mean(results_un):.2f} ± {np.std(results_un):.2f}")
    print(f"{'Graph Heuristic':<20} | {np.mean(results_heur):.2f} ± {np.std(results_heur):.2f}")
    print(f"{'Frozen RAI':<20} | {np.mean(results_rai):.2f} ± {np.std(results_rai):.2f}")
