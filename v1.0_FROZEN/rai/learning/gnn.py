import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv
import torch.nn.functional as F

class GraphEncoder(nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                ('entity', 'inputs', 'relation'): SAGEConv((-1, -1), hidden_channels),
                ('relation', 'outputs', 'entity'): SAGEConv((-1, -1), hidden_channels),
                # We also want message passing backwards to allow full information flow
                ('relation', 'rev_inputs', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'rev_outputs', 'relation'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
            
        # Linear layers for initial projection (since inputs are just [1.0])
        self.lin_dict = nn.ModuleDict({
            'entity': nn.Linear(1, hidden_channels),
            'relation': nn.Linear(1, hidden_channels)
        })

    def forward(self, x_dict, edge_index_dict):
        # Initial projection
        h_dict = {
            node_type: F.relu(self.lin_dict[node_type](x)) 
            for node_type, x in x_dict.items()
        }
        
        for conv in self.convs:
            out_dict = conv(h_dict, edge_index_dict)
            h_dict = {
                node_type: F.relu(out_dict[node_type])
                for node_type in out_dict.keys()
            }
            
        return h_dict


class CandidateScorer(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        # Takes in: source_entity_emb (hidden_channels), target_entity_emb (hidden_channels)
        # We assume 1 input entity and 1 output entity for now to keep it simple for scoring
        # If multi-input/output, we can mean-pool the input entities and output entities
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, src_emb, dst_emb):
        """
        src_emb: (batch_size, hidden_channels)
        dst_emb: (batch_size, hidden_channels)
        Returns: logits of shape (batch_size, 1)
        """
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.mlp(combined)

class RAIGNN(nn.Module):
    def __init__(self, hidden_channels: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = GraphEncoder(hidden_channels, num_layers)
        self.scorer = CandidateScorer(hidden_channels)
        
    def forward(self, x_dict, edge_index_dict, candidate_src_indices, candidate_dst_indices):
        """
        x_dict, edge_index_dict: from HeteroData (visible graph)
        candidate_src_indices: tensor of entity indices that are sources of candidate relation
        candidate_dst_indices: tensor of entity indices that are targets of candidate relation
        """
        h_dict = self.encoder(x_dict, edge_index_dict)
        
        ent_embs = h_dict['entity']
        
        src_embs = ent_embs[candidate_src_indices]
        dst_embs = ent_embs[candidate_dst_indices]
        
        return self.scorer(src_embs, dst_embs)
