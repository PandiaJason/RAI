from typing import Dict, Any
from rai.core.relation import Relation
from rai.core.agent import Agent

def create_transform_action(relation_id: int) -> Dict[str, Any]:
    """
    Creates an action dictionary for an agent to execute a transformation.
    """
    return {
        "type": "TRANSFORM",
        "relation_id": relation_id
    }
