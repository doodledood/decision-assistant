import enum
import json
from typing import Any, Optional

from halo import Halo
from pydantic.v1 import BaseModel, Field

from main import mark_stage_as_done


class Stage(enum.IntEnum):
    GOAL_IDENTIFICATION = 0
    ALTERNATIVE_LISTING = 1
    CRITERIA_IDENTIFICATION = 2
    CRITERIA_MAPPING = 3
    CRITERIA_PRIORITIZATION = 4
    CRITERIA_RESEARCH_QUESTIONS_GENERATION = 5
    DATA_RESEARCH = 6
    DATA_ANALYSIS = 7
    PRESENTATION_COMPILATION = 8


class DecisionAssistantState(BaseModel):
    last_completed_stage: Optional[Stage] = Field(description='The current stage of the decision-making process.')
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


def save_and_mark_stage_as_done(state: DecisionAssistantState, state_file: Optional[str]):
    with Halo(text='Saving state...', spinner='dots') as spinner:
        save_state(state, state_file)

        mark_stage_as_done(state.last_completed_stage, spinner)


def mark_stage_as_done(stage: Stage, halo: Optional[Halo] = None):
    if halo is None:
        halo = Halo(spinner='dots')

    stage_text = {
        Stage.GOAL_IDENTIFICATION: 'Goal identified.',
        Stage.ALTERNATIVE_LISTING: 'Alternatives listed.',
        Stage.CRITERIA_IDENTIFICATION: 'Criteria identified.',
        Stage.CRITERIA_MAPPING: 'Criteria mapped.',
        Stage.CRITERIA_PRIORITIZATION: 'Criteria prioritized.',
        Stage.CRITERIA_RESEARCH_QUESTIONS_GENERATION: 'Research questions generated.',
        Stage.DATA_RESEARCH: 'Data researched.',
        Stage.DATA_ANALYSIS: 'Data analyzed.',
        Stage.PRESENTATION_COMPILATION: 'Presentation compiled.'
    }[stage]
    halo.succeed(stage_text)
