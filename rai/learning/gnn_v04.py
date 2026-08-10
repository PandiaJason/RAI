import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
import networkx as nx
import numpy as np

def build_candidate_relative_batch(data, max_dist=10):
    """
    Transforms a single HeteroData graph (with C candidates) into a batched HeteroData
    containing C disconnected copies of the graph.
    In the i-th copy, node features are structural labels [d(x, u_i), d(x, v_i)]
    where (u_i, v_i) is the i-th candidate link.
    """
    device = data['entity'].x.device
    num_entities = data['entity'].x.shape[0]
    num_relations = data['relation'].x.shape[0]
    
    # 1. Reconstruct X-X undirected graph for fast APSP
    x_to_r = data['entity', 'inputs', 'relation'].edge_index.t().cpu().numpy()
    r_to_x = data['relation', 'outputs', 'entity'].edge_index.t().cpu().numpy()
    
    r_to_src = {r: src for src, r in x_to_r}
    edges = []
    for r, dst in r_to_x:
        if r in r_to_src:
            edges.append((r_to_src[r], dst))
            
    G = nx.Graph()
    G.add_nodes_from(range(num_entities))
    G.add_edges_from(edges)
    
    # Precompute APSP
    # dict of dicts: dict(nx.all_pairs_shortest_path_length(G))
    apsp = dict(nx.all_pairs_shortest_path_length(G))
    
    # Distance lookup function
    def get_dist(src, dst):
        if src in apsp and dst in apsp[src]:
            return min(apsp[src][dst], max_dist)
        return max_dist

    # 2. Build the C copies
    candidates_src = data.candidate_src.cpu().numpy()
    candidates_dst = data.candidate_dst.cpu().numpy()
    labels = data.candidate_labels.cpu().numpy()
    
    copies = []
    for i in range(len(candidates_src)):
        u = candidates_src[i]
        v = candidates_dst[i]
        
        c_data = HeteroData()
        
        # Build relative entity features: [d(x, u), d(x, v)]
        ent_feat = torch.zeros((num_entities, 2), dtype=torch.float, device=device)
        for x in range(num_entities):
            ent_feat[x, 0] = get_dist(x, u)
            ent_feat[x, 1] = get_dist(x, v)
        c_data['entity'].x = ent_feat
        
        # Relation nodes just get constant feature [1.0]
        # (Message passing pulls structural information from connected entities)
        c_data['relation'].x = torch.ones((num_relations, 1), dtype=torch.float, device=device)
        
        # Copy edge indices
        for edge_type in data.edge_types:
            c_data[edge_type].edge_index = data[edge_type].edge_index.clone()
            
        # For this copy, we only have 1 candidate!
        c_data.candidate_src = torch.tensor([u], dtype=torch.long, device=device)
        c_data.candidate_dst = torch.tensor([v], dtype=torch.long, device=device)
        c_data.candidate_labels = torch.tensor([labels[i]], dtype=torch.float, device=device)
        
        copies.append(c_data)
        
    return Batch.from_data_list(copies)

class CandidateRelativeEncoder(nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                ('entity', 'inputs', 'relation'): SAGEConv((-1, -1), hidden_channels),
                ('relation', 'outputs', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('relation', 'rev_inputs', 'entity'): SAGEConv((-1, -1), hidden_channels),
                ('entity', 'rev_outputs', 'relation'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
            
            norm = nn.ModuleDict({
                'entity': nn.LayerNorm(hidden_channels),
                'relation': nn.LayerNorm(hidden_channels)
            })
            self.norms.append(norm)
            
        # entity features are 2D discrete distances (dist_u, dist_v)
        # We use an Embedding layer for distances 0..10
        self.dist_emb = nn.Embedding(11, hidden_channels // 2)
        
        self.lin_rel = nn.Linear(1, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        # x_dict['entity'] is (N, 2) LongTensor of distances
        # x_dict['relation'] is (N, 1) FloatTensor
        
        d_u = self.dist_emb(x_dict['entity'][:, 0].long())
        d_v = self.dist_emb(x_dict['entity'][:, 1].long())
        ent_h = torch.cat([d_u, d_v], dim=-1) # hidden_channels
        
        h_dict = {
            'entity': F.relu(ent_h),
            'relation': F.relu(self.lin_rel(x_dict['relation']))
        }
        
        for conv, norm in zip(self.convs, self.norms):
            out_dict = conv(h_dict, edge_index_dict)
            h_dict = {
                node_type: F.relu(norm[node_type](out_dict[node_type]))
                for node_type in out_dict.keys()
            }
            
        return h_dict

class CandidateScorer(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, src_emb, dst_emb):
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.mlp(combined)

class CandidateRelativeRAIGNN(nn.Module):
    def __init__(self, hidden_channels: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = CandidateRelativeEncoder(hidden_channels, num_layers)
        self.scorer = CandidateScorer(hidden_channels)
        
    def forward(self, batched_data):
        """
        Expects a PyG Batch returned by build_candidate_relative_batch
        """
        h_dict = self.encoder(batched_data.x_dict, batched_data.edge_index_dict)
        
        ent_embs = h_dict['entity']
        
        # In the batched graph, the entities are concatenated.
        # We need to extract the exact candidate_src and candidate_dst for EACH copy.
        # Batching offsets the indices automatically!
        # batched_data.candidate_src contains the global offset indices.
        
        src_embs = ent_embs[batched_data.candidate_src]
        dst_embs = ent_embs[batched_data.candidate_dst]
        
        return self.scorer(src_embs, dst_embs)
