from typing import List, Optional, Dict, Any, Callable, TypeVar, Type

from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, FunctionMessage
from pydantic import BaseModel

from chat.base import Chat, TOutputSchema
from chat.conductors import RoundRobinChatConductor
from chat.errors import FunctionNotFoundError, MessageCouldNotBeParsedError
from chat.participants import LangChainBasedAIChatParticipant
from chat.participants.output_parser import JSONOutputParserChatParticipant
from chat.utils import pydantic_to_json_schema


def execute_chat_model_messages(chat_model: BaseChatModel,
                                messages: List[BaseMessage],
                                chat_model_args: Optional[Dict[str, Any]] = None,
                                functions: Optional[Dict[str, Callable[[Any], str]]] = None,
                                spinner: Optional[Halo] = None) -> str:
    chat_model_args = chat_model_args or {}
    functions = functions or {}

    all_messages = messages.copy()

    last_message = chat_model.predict_messages(all_messages, **chat_model_args)
    function_call = last_message.additional_kwargs.get('function_call')

    while function_call is not None:
        function_name = function_call['name']
        if function_name in functions:
            args = function_call['arguments']

            if spinner is not None:
                progress_text = f'Executing function `{function_name}`...'

                spinner.start(progress_text)

            function = functions[function_name]

            result = function(args)

            all_messages.append(FunctionMessage(
                name=function_name,
                content=result or 'None'
            ))

            last_message = chat_model.predict_messages(all_messages, **chat_model_args)
            function_call = last_message.additional_kwargs.get('function_call')
        else:
            raise FunctionNotFoundError(function_name)

    return last_message.content


PydanticType = TypeVar('PydanticType', bound=Type[BaseModel])


def pydantic_to_openai_function(pydantic_type: PydanticType, function_name: Optional[str] = None,
                                function_description: Optional[str] = None) -> Dict:
    base_schema = pydantic_type.model_json_schema()
    description = function_description if function_description is not None else (pydantic_type.__doc__ or '')

    return {
        'name': function_name or pydantic_type.__name__,
        'description': description,
        'parameters': base_schema
    }


def string_output_to_pydantic(output: str,
                              chat_model: BaseChatModel,
                              output_schema: Type[TOutputSchema],
                              spinner: Optional[Halo] = None,
                              n_retries: int = 3,
                              hide_message: bool = True) -> TOutputSchema:
    text_to_json_ai = LangChainBasedAIChatParticipant(
        chat_model=chat_model,
        name='Text to JSON Converter',
        role='Text to JSON Converter',
        personal_mission='You will be provided some TEXT and a JSON SCHEMA. Your only mission is to convert the TEXT '
                         'to a JSON that follows the JSON SCHEMA provided. Your message should include only correct JSON.',
        spinner=spinner
    )
    json_parser = JSONOutputParserChatParticipant(output_schema=output_schema)

    parser_chat = Chat(
        initial_participants=[json_parser, text_to_json_ai],
        hide_messages=hide_message,
        max_total_messages=n_retries * 2
    )
    conductor = RoundRobinChatConductor()

    _ = conductor.initiate_chat_with_result(
        chat=parser_chat,
        initial_message=f'# TEXT\n{output}\n\n# JSON SCHEMA\n{pydantic_to_json_schema(output_schema)}'
    )

    if json_parser.output is None:
        raise MessageCouldNotBeParsedError(output)

    return json_parser.output
