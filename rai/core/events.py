import json
from typing import Dict, Any

class EventLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        # Clear the file on init
        with open(self.filepath, 'w') as f:
            pass

    def log_event(self, event_data: Dict[str, Any]):
        """Append a JSONL event to the log file."""
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(event_data) + '\n')
