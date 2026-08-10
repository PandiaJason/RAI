from typing import Dict, Any

def create_exchange_action(target_agent_id: int, give_entity_id: int, give_amount: float, receive_entity_id: int, receive_amount: float) -> Dict[str, Any]:
    """
    Creates an action dictionary for an agent to propose/execute an exchange.
    """
    return {
        "type": "EXCHANGE",
        "target_agent": target_agent_id,
        "give_entity": give_entity_id,
        "give_amount": give_amount,
        "receive_entity": receive_entity_id,
        "receive_amount": receive_amount
    }
