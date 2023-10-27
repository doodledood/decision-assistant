from typing import List, Optional, Dict, Any, Callable, TypeVar, Type

from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, FunctionMessage
from pydantic import BaseModel

from chat.errors import FunctionNotFoundError


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


def pydantic_to_openai_function(function_name: str, pydantic_type: PydanticType) -> Dict:
    base_schema = pydantic_type.model_json_schema()
    description = pydantic_type.__doc__ or ''

    return {
        'name': function_name,
        'description': description,
        'parameters': base_schema
    }
