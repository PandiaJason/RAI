import pytest
import numpy as np
from rai.emergence.inequality import calculate_gini
from rai.emergence.exchange_network import calculate_network_centrality
import networkx as nx

def test_gini_calculation():
    # Perfectly equal distribution
    equal = np.array([10.0, 10.0, 10.0])
    assert pytest.approx(calculate_gini(equal), 0.01) == 0.0
    
    # Very unequal
    unequal = np.array([0.0, 0.0, 100.0])
    # Gini of [0, 0, 100] is around 0.66
    assert calculate_gini(unequal) > 0.5

def test_network_centrality():
    G = nx.DiGraph()
    assert calculate_network_centrality(G) == 0.0
    
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    # Node 1 has degree 2 (out), out of 2 possible other nodes. 
    # Degree centrality should be 1.0 (2/2) or similar depending on nx directed definition
    # Actually nx.degree_centrality sums in and out degree, so node 1 has 2/2 = 1.0
    # Node 2 has 1/2 = 0.5
    # Max is 1.0
    assert pytest.approx(calculate_network_centrality(G), 0.01) == 1.0
