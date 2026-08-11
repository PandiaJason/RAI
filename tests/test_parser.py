import pytest
import tempfile
import os
from rai.generation.real_world_parser import RealWorldParser

def test_real_world_parser():
    # Create temp csv
    csv_content = "RelationName,Inputs,Outputs\nTestRel,Iron:2,Steel:1"
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as f:
        f.write(csv_content)
        temp_path = f.name
        
    try:
        parser = RealWorldParser()
        world = parser.parse_csv(temp_path, num_agents=10)
        
        # Verify semantics stripped
        assert len(parser.entity_map) == 2
        assert "Iron" in parser.entity_map
        assert "Steel" in parser.entity_map
        
        # Verify World built abstractly
        assert len(world.agents) == 10
        rels = world.hypergraph.get_all_relations()
        assert len(rels) == 1
        
        inputs = rels[0].inputs
        # Iron should have ID 0, Steel ID 1 based on mapping order
        iron_id = parser.entity_map["Iron"]
        assert list(inputs.keys())[0].id == iron_id
        assert list(inputs.values())[0] == 2.0
    finally:
        os.remove(temp_path)
