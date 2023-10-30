import json
from typing import Any, Optional

from pydantic import BaseModel, Field


class DecisionAssistantState(BaseModel):
    data: Any = Field(description='The data collected so far.')


def save_state(state: DecisionAssistantState, state_file: Optional[str]):
    if state_file is None:
        return

    data = state.dict()
    with open(state_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_state(state_file: Optional[str]) -> Optional[DecisionAssistantState]:
    if state_file is None:
        return None

    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
            return DecisionAssistantState.parse_obj(data)
    except FileNotFoundError:
        return None
