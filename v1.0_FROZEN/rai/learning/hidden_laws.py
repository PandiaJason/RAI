import random
import numpy as np
import networkx as nx

class HiddenLawGenerator:
    """
    Generates hidden relational laws for Entity-to-Entity connections.
    """
    
    @staticmethod
    def _generate_hubs(num_entities, num_edges):
        """Family A: Preferential Attachment (Hubs)"""
        m = max(1, num_edges // num_entities)
        if m >= num_entities: m = num_entities - 1
        g = nx.barabasi_albert_graph(n=num_entities, m=m)
        edges = []
        for u, v in g.edges():
            if random.random() > 0.5:
                edges.append((u, v))
            else:
                edges.append((v, u))
        return edges

    @staticmethod
    def _generate_communities(num_entities, num_edges):
        """Family B: Latent Communities"""
        num_communities = max(2, random.randint(2, max(3, num_entities // 5)))
        # Hidden latent groups
        communities = {i: random.randint(0, num_communities - 1) for i in range(num_entities)}
        
        # Compatibility rules (e.g. comm 0 connects to comm 1, etc.)
        # We define a hidden transition matrix between communities
        transition_prob = np.random.uniform(0, 1, (num_communities, num_communities))
        # Make diagonal (intra-community) higher probability usually, but not always
        for i in range(num_communities):
            transition_prob[i, i] = random.uniform(0.5, 1.0)
            
        edges = set()
        attempts = 0
        while len(edges) < num_edges and attempts < num_edges * 10:
            attempts += 1
            u = random.randint(0, num_entities - 1)
            v = random.randint(0, num_entities - 1)
            if u == v: continue
            
            c_u = communities[u]
            c_v = communities[v]
            
            if random.random() < transition_prob[c_u, c_v]:
                edges.add((u, v))
                
        return list(edges)

    @staticmethod
    def _generate_chains(num_entities, num_edges):
        """Family C: Compositional Chains"""
        edges = set()
        nodes = list(range(num_entities))
        random.shuffle(nodes)
        
        idx = 0
        while idx < num_entities - 1 and len(edges) < num_edges:
            chain_length = random.randint(3, max(4, num_entities // 3))
            if idx + chain_length > num_entities:
                chain_length = num_entities - idx
            
            for i in range(chain_length - 1):
                edges.add((nodes[idx + i], nodes[idx + i + 1]))
                if len(edges) >= num_edges: break
            idx += chain_length
            
        attempts = 0
        while len(edges) < num_edges and attempts < num_edges * 5:
            attempts += 1
            u = random.randint(0, num_entities - 1)
            v = random.randint(0, num_entities - 1)
            if u != v:
                edges.add((u, v))
                
        return list(edges)

    @staticmethod
    def _generate_dag(num_entities, num_edges):
        """Family D: Directed Acyclic Worlds"""
        # Latent topological ordering
        nodes = list(range(num_entities))
        random.shuffle(nodes) # nodes[0] before nodes[1] etc
        order = {node: i for i, node in enumerate(nodes)}
        
        edges = set()
        attempts = 0
        while len(edges) < num_edges and attempts < num_edges * 10:
            attempts += 1
            u = random.randint(0, num_entities - 1)
            v = random.randint(0, num_entities - 1)
            if u == v: continue
            
            if order[u] < order[v]:
                edges.add((u, v))
        return list(edges)

    @staticmethod
    def _generate_cycles(num_entities, num_edges):
        """Family E: Feedback Worlds"""
        edges = set()
        nodes = list(range(num_entities))
        random.shuffle(nodes)
        
        idx = 0
        while idx < num_entities - 2 and len(edges) < num_edges:
            cycle_length = random.randint(3, max(4, num_entities // 2))
            if idx + cycle_length > num_entities:
                cycle_length = num_entities - idx
            
            if cycle_length < 3: break
                
            for i in range(cycle_length - 1):
                edges.add((nodes[idx + i], nodes[idx + i + 1]))
            edges.add((nodes[idx + cycle_length - 1], nodes[idx]))
            idx += cycle_length
            
        attempts = 0
        while len(edges) < num_edges and attempts < num_edges * 5:
            attempts += 1
            u = random.randint(0, num_entities - 1)
            v = random.randint(0, num_entities - 1)
            if u != v: edges.add((u, v))
            
        return list(edges)
        
    @staticmethod
    def _generate_null(num_entities, num_edges):
        """Null Family: Erdős-Rényi random graphs"""
        edges = set()
        attempts = 0
        while len(edges) < num_edges and attempts < num_edges * 5:
            attempts += 1
            u = random.randint(0, num_entities - 1)
            v = random.randint(0, num_entities - 1)
            if u != v: edges.add((u, v))
        return list(edges)

    @staticmethod
    def generate(families: list, num_entities: int, num_edges: int):
        """
        Supports compositional generation by generating edges for each requested family
        and taking the union, capping at num_edges.
        """
        all_edges = set()
        edges_per_family = max(1, num_edges // len(families))
        
        for fam in families:
            fam = fam.upper()
            if fam == 'A': cur_edges = HiddenLawGenerator._generate_hubs(num_entities, edges_per_family)
            elif fam == 'B': cur_edges = HiddenLawGenerator._generate_communities(num_entities, edges_per_family)
            elif fam == 'C': cur_edges = HiddenLawGenerator._generate_chains(num_entities, edges_per_family)
            elif fam == 'D': cur_edges = HiddenLawGenerator._generate_dag(num_entities, edges_per_family)
            elif fam == 'E': cur_edges = HiddenLawGenerator._generate_cycles(num_entities, edges_per_family)
            elif fam == 'NULL': cur_edges = HiddenLawGenerator._generate_null(num_entities, edges_per_family)
            else:
                raise ValueError(f"Unknown family: {fam}")
            
            for e in cur_edges:
                all_edges.add(e)
                
        # If we need more edges to hit num_edges exactly (due to overlap or rounding)
        # we pad with completely random edges (Null) to hit density targets
        attempts = 0
        while len(all_edges) < num_edges and attempts < num_edges * 5:
            attempts += 1
            u = random.randint(0, num_entities - 1)
            v = random.randint(0, num_entities - 1)
            if u != v: all_edges.add((u, v))
            
        return list(all_edges)[:num_edges]
