from typing import Dict, Any

def create_explore_action() -> Dict[str, Any]:
    """
    Creates an action dictionary for an agent to attempt discovery.
    """
    return {
        "type": "EXPLORE"
    }
