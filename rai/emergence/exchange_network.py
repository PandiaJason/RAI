import json
import networkx as nx

def build_exchange_network(events_file: str) -> nx.DiGraph:
    """
    Builds a directed graph of exchanges between agents.
    """
    G = nx.DiGraph()
    
    with open(events_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            if data['event'] == 'EXCHANGE':
                a1 = data['agent_1']
                a2 = data['agent_2']
                
                # Add nodes if they don't exist
                if not G.has_node(a1): G.add_node(a1)
                if not G.has_node(a2): G.add_node(a2)
                
                # Add/update edges
                if G.has_edge(a1, a2):
                    G[a1][a2]['weight'] += 1
                else:
                    G.add_edge(a1, a2, weight=1)
                    
    return G

def calculate_network_centrality(G: nx.DiGraph) -> float:
    """
    Returns the maximum degree centrality to detect if exchange hubs emerged.
    """
    if len(G) == 0:
        return 0.0
    centrality = nx.degree_centrality(G)
    return max(centrality.values())
