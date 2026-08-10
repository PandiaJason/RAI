import torch
from torch_geometric.data import InMemoryDataset, HeteroData
import random
import os
from rai.learning.hidden_laws import HiddenLawGenerator

class SyntheticLinkPredictionDataset(InMemoryDataset):
    def __init__(self, root, composition_list=[['A'], ['B'], ['C']], num_graphs=1000, min_entities=10, max_entities=50, noise_ratio=0.0, transform=None, pre_transform=None):
        self.composition_list = composition_list
        self.num_graphs = num_graphs
        self.min_entities = min_entities
        self.max_entities = max_entities
        self.noise_ratio = noise_ratio
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []
        
    @property
    def processed_file_names(self):
        c_str = ""
        for c in self.composition_list:
            c_str += "".join(sorted(c)) + "_"
        return [f'data_{c_str}{self.num_graphs}_{self.min_entities}_{self.max_entities}_{self.noise_ratio}.pt']
        
    def download(self):
        pass
        
    def process(self):
        data_list = []
        for i in range(self.num_graphs):
            num_e = random.randint(self.min_entities, self.max_entities)
            max_possible_edges = num_e * (num_e - 1)
            
            target_edges = random.randint(int(num_e * 1.5), int(num_e * 4))
            target_edges = min(target_edges, max_possible_edges)
            
            families = random.choice(self.composition_list)
            all_edges = HiddenLawGenerator.generate(families, num_e, target_edges)
            
            # Add noise if requested
            if self.noise_ratio > 0:
                num_noise = int(len(all_edges) * self.noise_ratio)
                all_edges_set = set(all_edges)
                attempts = 0
                while len(all_edges) < len(all_edges_set) + num_noise and attempts < num_noise * 10:
                    attempts += 1
                    u = random.randint(0, num_e - 1)
                    v = random.randint(0, num_e - 1)
                    if u != v and (u, v) not in all_edges_set:
                        all_edges.append((u, v))
                        all_edges_set.add((u, v))
            
            all_edges = list(set(all_edges))
            random.shuffle(all_edges)
            
            num_hidden = max(1, int(len(all_edges) * 0.2))
            
            pos_edges = all_edges[:num_hidden]
            vis_edges = all_edges[num_hidden:]
            
            neg_edges = set()
            max_neg = max_possible_edges - len(all_edges)
            num_hidden = min(num_hidden, max_neg)
            
            attempts = 0
            while len(neg_edges) < num_hidden and attempts < num_hidden * 10:
                attempts += 1
                src = random.randint(0, num_e - 1)
                dst = random.randint(0, num_e - 1)
                if src != dst and (src, dst) not in all_edges:
                    neg_edges.add((src, dst))
            neg_edges = list(neg_edges)
            
            data = HeteroData()
            data['entity'].x = torch.ones((num_e, 1), dtype=torch.float)
            data['relation'].x = torch.ones((len(vis_edges), 1), dtype=torch.float)
            
            edge_index_X_to_R = []
            edge_index_R_to_X = []
            
            for rel_idx, (src, dst) in enumerate(vis_edges):
                edge_index_X_to_R.append([src, rel_idx])
                edge_index_R_to_X.append([rel_idx, dst])
                
            if len(vis_edges) > 0:
                data['entity', 'inputs', 'relation'].edge_index = torch.tensor(edge_index_X_to_R, dtype=torch.long).t().contiguous()
                data['relation', 'outputs', 'entity'].edge_index = torch.tensor(edge_index_R_to_X, dtype=torch.long).t().contiguous()
                data['relation', 'rev_inputs', 'entity'].edge_index = torch.tensor(edge_index_X_to_R, dtype=torch.long).t().contiguous().flip([0])
                data['entity', 'rev_outputs', 'relation'].edge_index = torch.tensor(edge_index_R_to_X, dtype=torch.long).t().contiguous().flip([0])
            else:
                empty = torch.empty((2, 0), dtype=torch.long)
                data['entity', 'inputs', 'relation'].edge_index = empty
                data['relation', 'outputs', 'entity'].edge_index = empty
                data['relation', 'rev_inputs', 'entity'].edge_index = empty
                data['entity', 'rev_outputs', 'relation'].edge_index = empty

            all_cands = pos_edges + neg_edges
            labels = [1]*len(pos_edges) + [0]*len(neg_edges)
            
            cands_and_labels = list(zip(all_cands, labels))
            random.shuffle(cands_and_labels)
            
            srcs = [c[0][0] for c in cands_and_labels]
            dsts = [c[0][1] for c in cands_and_labels]
            lbls = [c[1] for c in cands_and_labels]
            
            data.candidate_src = torch.tensor(srcs, dtype=torch.long)
            data.candidate_dst = torch.tensor(dsts, dtype=torch.long)
            data.candidate_labels = torch.tensor(lbls, dtype=torch.float)
            
            data_list.append(data)
            
        self.save(data_list, self.processed_paths[0])
