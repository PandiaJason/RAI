import torch
import random
from torch_geometric.data import HeteroData
from rai.learning.gnn import RAIGNN
import numpy as np

def test_permutation_equivariance():
    # 1. Initialize random graph
    num_e = 15
    num_r = 30
    
    data = HeteroData()
    data['entity'].x = torch.ones((num_e, 1), dtype=torch.float)
    data['relation'].x = torch.ones((num_r, 1), dtype=torch.float)
    
    e_to_r = []
    r_to_e = []
    
    for r in range(num_r):
        src = random.randint(0, num_e - 1)
        dst = random.randint(0, num_e - 1)
        e_to_r.append([src, r])
        r_to_e.append([r, dst])
        
    data['entity', 'inputs', 'relation'].edge_index = torch.tensor(e_to_r, dtype=torch.long).t().contiguous()
    data['relation', 'outputs', 'entity'].edge_index = torch.tensor(r_to_e, dtype=torch.long).t().contiguous()
    data['relation', 'rev_inputs', 'entity'].edge_index = torch.tensor(e_to_r, dtype=torch.long).t().contiguous().flip([0])
    data['entity', 'rev_outputs', 'relation'].edge_index = torch.tensor(r_to_e, dtype=torch.long).t().contiguous().flip([0])
    
    model = RAIGNN(hidden_channels=32, num_layers=2)
    model.eval()
    
    # Random candidates
    cand_src = torch.tensor([0, 5, 12], dtype=torch.long)
    cand_dst = torch.tensor([1, 8, 3], dtype=torch.long)
    
    with torch.no_grad():
        out_original = model(data.x_dict, data.edge_index_dict, cand_src, cand_dst)
        
    # 2. Permute node IDs
    # Create a mapping from old -> new
    mapping = list(range(num_e))
    random.shuffle(mapping)
    # reverse mapping new -> old
    rev_mapping = {new_id: old_id for old_id, new_id in enumerate(mapping)}
    
    perm_e_to_r = [[mapping[src], r] for src, r in e_to_r]
    perm_r_to_e = [[r, mapping[dst]] for r, dst in r_to_e]
    
    perm_data = HeteroData()
    perm_data['entity'].x = torch.ones((num_e, 1), dtype=torch.float)
    perm_data['relation'].x = torch.ones((num_r, 1), dtype=torch.float)
    
    perm_data['entity', 'inputs', 'relation'].edge_index = torch.tensor(perm_e_to_r, dtype=torch.long).t().contiguous()
    perm_data['relation', 'outputs', 'entity'].edge_index = torch.tensor(perm_r_to_e, dtype=torch.long).t().contiguous()
    perm_data['relation', 'rev_inputs', 'entity'].edge_index = torch.tensor(perm_e_to_r, dtype=torch.long).t().contiguous().flip([0])
    perm_data['entity', 'rev_outputs', 'relation'].edge_index = torch.tensor(perm_r_to_e, dtype=torch.long).t().contiguous().flip([0])
    
    # Update candidate indices
    perm_cand_src = torch.tensor([mapping[x.item()] for x in cand_src], dtype=torch.long)
    perm_cand_dst = torch.tensor([mapping[x.item()] for x in cand_dst], dtype=torch.long)
    
    with torch.no_grad():
        out_permuted = model(perm_data.x_dict, perm_data.edge_index_dict, perm_cand_src, perm_cand_dst)
        
    # 3. Assert outputs are mathematically identical (order of candidates is preserved)
    diff = torch.abs(out_original - out_permuted).max().item()
    print(f"Max difference after full node ID permutation: {diff:.8f}")
    assert diff < 1e-5, f"Permutation equivariance failed! Diff: {diff}"
    print("SUCCESS: RAI-GNN v0.2 is perfectly permutation equivariant.")

if __name__ == "__main__":
    test_permutation_equivariance()
