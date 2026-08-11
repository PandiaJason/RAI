import random
import numpy as np
import networkx as nx

class HiddenLawGenerator:
    """
    Generates hidden relational laws for Entity-to-Entity connections.
    Later these are expanded into Bipartite X -> R -> X edges.
    """
    
    @staticmethod
    def generate_hubs(num_entities, num_edges):
        """Family A: Preferential Attachment (Hubs) - scale-free network"""
        # Barabasi-Albert model using networkx
        m = max(1, num_edges // num_entities)
        if m >= num_entities: m = num_entities - 1
        g = nx.barabasi_albert_graph(n=num_entities, m=m)
        edges = list(g.edges())
        # Make directed randomly
        directed_edges = []
        for u, v in edges:
            if random.random() > 0.5:
                directed_edges.append((u, v))
            else:
                directed_edges.append((v, u))
        return directed_edges

    @staticmethod
    def generate_communities(num_entities, num_edges):
        """Family B: Community Block Structure"""
        num_communities = max(2, num_entities // 10)
        communities = {i: random.randint(0, num_communities - 1) for i in range(num_entities)}
        
        edges = set()
        attempts = 0
        while len(edges) < num_edges and attempts < num_edges * 10:
            attempts += 1
            u = random.randint(0, num_entities - 1)
            v = random.randint(0, num_entities - 1)
            if u == v: continue
            
            same_comm = communities[u] == communities[v]
            # 80% chance for intra-community, 20% for inter-community
            if same_comm and random.random() < 0.8:
                edges.add((u, v))
            elif not same_comm and random.random() < 0.2:
                edges.add((u, v))
                
        return list(edges)

    @staticmethod
    def generate_chains(num_entities, num_edges):
        """Family C: Compositional Chains"""
        # Create disjoint paths/chains of random lengths
        edges = set()
        nodes = list(range(num_entities))
        random.shuffle(nodes)
        
        idx = 0
        while idx < num_entities - 1 and len(edges) < num_edges:
            chain_length = random.randint(2, 5)
            if idx + chain_length > num_entities:
                chain_length = num_entities - idx
                
            for i in range(chain_length - 1):
                edges.add((nodes[idx + i], nodes[idx + i + 1]))
                if len(edges) >= num_edges: break
            idx += chain_length
            
        # If we need more edges, add random cross-chain connections
        attempts = 0
        while len(edges) < num_edges and attempts < num_edges * 10:
            attempts += 1
            u = random.randint(0, num_entities - 1)
            v = random.randint(0, num_entities - 1)
            if u != v:
                edges.add((u, v))
                
        return list(edges)

    @staticmethod
    def generate_dag(num_entities, num_edges):
        """Family D: Bipartite DAGs (Acyclic)"""
        nodes = list(range(num_entities))
        # implicit topological sort is just the integers 0..N-1
        edges = set()
        attempts = 0
        while len(edges) < num_edges and attempts < num_edges * 10:
            attempts += 1
            u = random.randint(0, num_entities - 2)
            v = random.randint(u + 1, num_entities - 1)
            edges.add((u, v))
        return list(edges)

    @staticmethod
    def generate_cycles(num_entities, num_edges):
        """Family E: Feedback Cycles"""
        edges = set()
        nodes = list(range(num_entities))
        random.shuffle(nodes)
        
        idx = 0
        while idx < num_entities - 2 and len(edges) < num_edges:
            cycle_length = random.randint(3, 6)
            if idx + cycle_length > num_entities:
                cycle_length = num_entities - idx
            
            if cycle_length < 3: break
                
            for i in range(cycle_length - 1):
                edges.add((nodes[idx + i], nodes[idx + i + 1]))
            # close the cycle
            edges.add((nodes[idx + cycle_length - 1], nodes[idx]))
            idx += cycle_length
            
        attempts = 0
        while len(edges) < num_edges and attempts < num_edges * 10:
            attempts += 1
            u = random.randint(0, num_entities - 1)
            v = random.randint(0, num_entities - 1)
            if u != v: edges.add((u, v))
            
        return list(edges)

    @staticmethod
    def generate(family: str, num_entities: int, num_edges: int):
        family = family.upper()
        if family == 'A': return HiddenLawGenerator.generate_hubs(num_entities, num_edges)
        if family == 'B': return HiddenLawGenerator.generate_communities(num_entities, num_edges)
        if family == 'C': return HiddenLawGenerator.generate_chains(num_entities, num_edges)
        if family == 'D': return HiddenLawGenerator.generate_dag(num_entities, num_edges)
        if family == 'E': return HiddenLawGenerator.generate_cycles(num_entities, num_edges)
        raise ValueError(f"Unknown generative family: {family}")
