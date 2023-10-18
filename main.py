import enum
import json
from typing import Optional, List, Tuple, Any, Dict

from dotenv import load_dotenv
from fire import Fire
from halo import Halo
from langchain.callbacks import StreamingStdOutCallbackHandler, StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.vectorstores.chroma import Chroma
from pydantic.v1 import BaseModel, Field

import system_prompts
from chat import chat
from research import create_web_search_tool, WebSearch


class Criterion(BaseModel):
    name: str = Field(description='The name of the criterion. Example: "Affordability".')
    scale: List[str] = Field(
        description='The 5-point scale of the criterion, from worst to best. Labels only. No numerical value, no explainations. Example: "Very Expensive".')


class CriteriaIdentificationResult(BaseModel):
    criteria: List[Criterion] = Field(description='The identified criteria for evaluating the decision.')


class AlternativeListingResult(BaseModel):
    alternatives: List[str] = Field(description='The identified alternatives for the decision.')


class CriteriaMappingResult(BaseModel):
    criteria_mapping: List[str] = Field(
        description='An explaination for each criterion on how to assign a value from the scale to a piece of data. Ordered in the same way as the criteria.')


class CriteriaPrioritizationResult(BaseModel):
    criteria_weights: List[int] = Field(
        description='The weights of the criteria, from 1 to 100, reflecting their relative importance based on the user\'s preferences. Ordered in the same way as the criteria.')


class Stage(int, enum.Enum):
    GOAL_IDENTIFICATION = 0
    ALTERNATIVE_LISTING = 1
    CRITERIA_IDENTIFICATION = 2
    CRITERIA_MAPPING = 3
    CRITERIA_PRIORITIZATION = 4
    DATA_RESEARCH = 5
    PRESENTATION = 6


class DecisionAssistantState(BaseModel):
    last_completed_stage: Optional[Stage] = Field(description='The current stage of the decision-making process.')
    data: Any = Field(description='The data collected so far.')


class Alternative(BaseModel):
    name: str = Field(description='The name of the alternative.')
    criteria_data: Optional[Dict[str, Tuple[str, int]]] = Field(
        description='The research data collected for each criterion for this alternative. Key is the name of the criterion. Value is a tuple of the research data as text and the assigned value based on the 5-point scale of the criterion.')


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


def mark_stage_as_done(stage: Stage, halo: Optional[Halo] = None):
    if halo is None:
        halo = Halo(spinner='dots')

    stage_text = {
        Stage.GOAL_IDENTIFICATION: 'Goal identified.',
        Stage.ALTERNATIVE_LISTING: 'Alternatives listed.',
        Stage.CRITERIA_IDENTIFICATION: 'Criteria identified.',
        Stage.CRITERIA_MAPPING: 'Criteria mapped.',
        Stage.CRITERIA_PRIORITIZATION: 'Criteria prioritized.',
        Stage.DATA_RESEARCH: 'Data researched.',
        Stage.PRESENTATION: 'Presentation ready.',
    }[stage]
    halo.succeed(stage_text)


def save_and_mark_stage_as_done(state: DecisionAssistantState, state_file: Optional[str]):
    with Halo(text='Saving state...', spinner='dots') as spinner:
        save_state(state, state_file)

        mark_stage_as_done(state.last_completed_stage, spinner)


def run_decision_assistant(goal: Optional[str] = None, llm_temperature: float = 0.0, llm_model: str = 'gpt-4-0613',
                           state_file: Optional[str] = 'state.json', streaming: bool = False,
                           n_search_results: int = 5):
    chat_model = ChatOpenAI(temperature=llm_temperature, model=llm_model, streaming=streaming,
                            callbacks=[StreamingStdOutCallbackHandler() if streaming else StdOutCallbackHandler()])
    default_tools = [create_web_search_tool(search=WebSearch(
        chat_model=chat_model,
        vectorstore=Chroma(embedding_function=OpenAIEmbeddings()),
        search=GoogleSearchAPIWrapper(k=n_search_results)
    ))]

    with Halo(text='Loading previous state...', spinner='dots') as spinner:
        state = load_state(state_file)
        if state is None:
            state = DecisionAssistantState(stage=None, data={})
        else:
            spinner.succeed('Loaded previous state.')

    if state.last_completed_stage is None and goal is None:
        goal = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.goal_identification_system_prompt),
            HumanMessage(content="Hey")
        ], tools=default_tools)

        state.last_completed_stage = Stage.GOAL_IDENTIFICATION
        state.data = {**state.data, **dict(goal=goal)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.GOAL_IDENTIFICATION)

    goal = state.data['goal']
    if state.last_completed_stage == Stage.GOAL_IDENTIFICATION:
        alternatives = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.alternative_listing_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}'),
        ], tools=default_tools, result_schema=AlternativeListingResult)
        alternatives = alternatives.dict()['alternatives']

        state.last_completed_stage = Stage.ALTERNATIVE_LISTING
        state.data = {**state.data, **dict(alternatives=alternatives)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.ALTERNATIVE_LISTING)

    alternatives = state.data['alternatives']
    if state.last_completed_stage == Stage.ALTERNATIVE_LISTING:
        criteria = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_identification_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}'),
        ], tools=default_tools, result_schema=CriteriaIdentificationResult)
        criteria = criteria.dict()['criteria']

        state.last_completed_stage = Stage.CRITERIA_IDENTIFICATION
        state.data = {**state.data, **dict(criteria=criteria)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_IDENTIFICATION)

    criteria = state.data['criteria']
    if state.last_completed_stage == Stage.CRITERIA_IDENTIFICATION:
        criteria_mapping = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_mapping_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}\n\n# CRITERIA\n{criteria}'),
        ], tools=default_tools, result_schema=CriteriaMappingResult)
        criteria_mapping = criteria_mapping.dict()['criteria_mapping']

        state.last_completed_stage = Stage.CRITERIA_MAPPING
        state.data = {**state.data, **dict(criteria_mapping=criteria_mapping)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_MAPPING)

    criteria_mapping = state.data['criteria_mapping']
    if state.last_completed_stage == Stage.CRITERIA_MAPPING:
        criteria_weights = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_prioritization_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}\n\n# CRITERIA\n{criteria}\n\n# CRITERIA MAPPING\n{criteria_mapping}'),
        ], tools=default_tools, result_schema=CriteriaPrioritizationResult)
        criteria_weights = criteria_weights.dict()['criteria_weights']

        state.last_completed_stage = Stage.CRITERIA_PRIORITIZATION
        state.data = {**state.data, **dict(criteria_weights=criteria_weights)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_PRIORITIZATION)

    criteria_weights = state.data['criteria_weights']

    print(state)


if __name__ == '__main__':
    load_dotenv()

    Fire(run_decision_assistant)
