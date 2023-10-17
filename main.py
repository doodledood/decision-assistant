import enum
import json
from typing import Optional, List, Callable, TypeVar, Type, Tuple, Any, Dict, Union

from dotenv import load_dotenv
from fire import Fire
from halo import Halo
from langchain.callbacks import StreamingStdOutCallbackHandler, StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, FunctionMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from pydantic.v1 import BaseModel, Field

import system_prompts

ResultSchema = TypeVar("T", bound=BaseModel)


def pydantic_to_json_schema(pydantic_model: Type[BaseModel]) -> dict:
    try:
        return pydantic_model.model_json_schema()
    except AttributeError:
        return pydantic_model.schema()


def json_string_to_pydantic(json_string: str, pydantic_model: Type[BaseModel]) -> BaseModel:
    try:
        return pydantic_model.model_validate_json(json_string)
    except AttributeError:
        return pydantic_model.parse_raw(json_string)


def chat(chat_model: ChatOpenAI,
         messages: List[BaseMessage],
         tools: Optional[List[Tool]] = None,
         get_user_input: Optional[Callable[[List[BaseMessage]], str]] = None,
         on_reply: Optional[Callable[[str], None]] = None,
         result_schema: Optional[Type[BaseModel]] = None,
         use_halo: bool = True
         ) -> ResultSchema:
    assert len(messages) > 0, 'At least one message is required.'

    if get_user_input is None:
        get_user_input = lambda _: input('\n> ')

    if on_reply is None:
        on_reply = lambda x: None

    all_messages = messages
    functions = (
            [{
                "name": '_terminate',
                "description": 'Should be called when you think you have achieved your mission or goal and ready to move on to the next step. The result of the mission or goal should be provided as an argument.',
                "parameters": {
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "The result of the mission or goal."
                        } if result_schema is None else pydantic_to_json_schema(result_schema),
                    },
                    "required": ["result"],
                    "type": "object"
                }
            }] +
            [format_tool_to_openai_function(tool) for tool in (tools or [])]
    )

    while True:
        if use_halo:
            spinner = Halo(text='Thinking...', spinner='dots')
            spinner.start()

            last_message = chat_model.predict_messages(all_messages, functions=functions)

            if last_message.content != '':
                spinner.stop_and_persist(symbol='🤖', text=last_message.content)
        else:
            last_message = chat_model.predict_messages(all_messages, functions=functions)

        all_messages.append(last_message)

        function_call = last_message.additional_kwargs.get('function_call')
        if function_call is not None:
            if function_call['name'] == '_terminate':
                args = json.loads(function_call['arguments'])

                result = args['result']
                if result_schema is not None:
                    if isinstance(result, str):
                        result = json_string_to_pydantic(str(result), result_schema)
                    else:
                        result = result_schema.parse_obj(result)

                return result

            for tool in tools:
                if tool.name == function_call['name']:
                    orig_args = args = function_call['arguments']
                    progress_text = 'Executing function call...'

                    if tool.args_schema is not None and isinstance(args, str):
                        args = json_string_to_pydantic(args, tool.args_schema)
                        if isinstance(args, str):
                            args = json_string_to_pydantic(str(args), tool.args_schema)
                        else:
                            args = tool.args_schema.parse_obj(args)

                        if hasattr(args, 'progress_text'):
                            progress_text = args.progress_text

                    with Halo(text=progress_text, spinner='dots'):
                        result = tool.run(orig_args)
                        messages.append(FunctionMessage(
                            name=tool.name,
                            content=result or 'None'
                        ))

                    break
        else:
            last_message = messages[-1]
            on_reply(last_message.content)

            user_input = get_user_input(messages)
            all_messages.append(HumanMessage(content=user_input))


class Criterion(BaseModel):
    name: str = Field(description='The name of the criterion.')
    scale: List[str] = Field(
        description='The 5-point scale of the criterion, from worst to best. Numerical values of the scale should be excluded, only label included.')


class CriterionWithMapping(Criterion):
    mapping_explanation: str = Field(
        description='The concrete description of how to assign a value form the scale to a piece of data.')


class CriterionWithWeight(CriterionWithMapping):
    weight: int = Field(description='The weight of the criterion, from 1 to 100, reflecting its relative importance.')


class CriteriaIdentificationResult(BaseModel):
    criteria: List[Criterion] = Field(description='The identified criteria for evaluating the decision.')


class CriteriaMappingResult(BaseModel):
    criteria: List[CriterionWithMapping] = Field(
        description='The identified criteria for evaluating the decision with concrete mappings and explainations for how to assign values.')


class CriteriaPrioritizationResult(BaseModel):
    criteria: List[CriterionWithWeight] = Field(
        description='The identified criteria for evaluating the decision with concrete mappings, explainations for how to assign values, and weights reflecting their relative importance.')


class Stage(int, enum.Enum):
    GOAL_IDENTIFICATION = 0
    CRITERIA_IDENTIFICATION = 1
    CRITERIA_MAPPING = 2
    CRITERIA_PRIORITIZATION = 3
    DATA_RESEARCH = 4
    PRESENTATION = 5


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
        Stage.GOAL_IDENTIFICATION: 'Goal identified',
        Stage.CRITERIA_IDENTIFICATION: 'Criteria identified',
        Stage.CRITERIA_MAPPING: 'Criteria mapped',
        Stage.CRITERIA_PRIORITIZATION: 'Criteria prioritized',
        Stage.DATA_RESEARCH: 'Data researched',
        Stage.PRESENTATION: 'Presentation ready',
    }[stage]
    halo.succeed(stage_text)


def save_and_mark_stage_as_done(state: DecisionAssistantState, state_file: Optional[str]):
    with Halo(text='Saving state...', spinner='dots') as spinner:
        save_state(state, state_file)

        mark_stage_as_done(state.last_completed_stage, spinner)


def run_decision_assistant(goal: Optional[str] = None, llm_temperature: float = 0.0, llm_model: str = 'gpt-4-0613',
                           state_file: Optional[str] = 'state.json', streaming: bool = False):
    chat_model = ChatOpenAI(temperature=llm_temperature, model=llm_model, streaming=streaming,
                            callbacks=[StreamingStdOutCallbackHandler() if streaming else StdOutCallbackHandler()])

    with Halo(text='Loading previous state...', spinner='dots') as spinner:
        state = load_state(state_file)
        if state is None:
            state = DecisionAssistantState(stage=None, data=None)
        else:
            spinner.succeed('Loaded previous state.')

    if state.last_completed_stage is None and goal is None:
        goal = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.goal_identification_system_prompt),
            HumanMessage(content="Hey")
        ])

        state.last_completed_stage = Stage.GOAL_IDENTIFICATION
        state.data = dict(goal=goal)

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.GOAL_IDENTIFICATION)

    goal = state.data['goal']
    if state.last_completed_stage == Stage.GOAL_IDENTIFICATION:
        criteria = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_identification_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}'),
        ], result_schema=CriteriaIdentificationResult)
        criteria = criteria.dict()['criteria']

        state.last_completed_stage = Stage.CRITERIA_IDENTIFICATION
        state.data = dict(goal=goal, criteria=criteria)

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_IDENTIFICATION)

    criteria = state.data['criteria']
    if state.last_completed_stage == Stage.CRITERIA_IDENTIFICATION:
        criteria_with_mapping = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_mapping_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}\n\n# CRITERIA\n{criteria}'),
        ], result_schema=CriteriaMappingResult)
        criteria_with_mapping = criteria_with_mapping.dict()['criteria']

        state.last_completed_stage = Stage.CRITERIA_MAPPING
        state.data = dict(goal=goal, criteria=criteria_with_mapping)

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_MAPPING)

    criteria_with_mapping = state.data['criteria']
    if state.last_completed_stage == Stage.CRITERIA_MAPPING:
        criteria_with_mapping_and_prioritization = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_prioritization_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}\n\n# CRITERIA: {criteria_with_mapping}'),
        ], result_schema=CriteriaMappingResult)
        criteria_with_mapping_and_prioritization = criteria_with_mapping_and_prioritization.dict()['criteria']

        state.last_completed_stage = Stage.CRITERIA_PRIORITIZATION
        state.data = dict(goal=goal, criteria=criteria_with_mapping_and_prioritization)

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_PRIORITIZATION)

    print(state)


if __name__ == '__main__':
    load_dotenv()

    Fire(run_decision_assistant)
