import dataclasses
from typing import List, Optional, Dict, Any, Callable, TypeVar, Type, Union, Tuple

from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, FunctionMessage
from pydantic import BaseModel

from chat.errors import FunctionNotFoundError
from chat.utils import json_string_to_pydantic

TFunctionArgsSchema = TypeVar('TFunctionArgsSchema', bound=Union[Type[BaseModel], Type[str]])


@dataclasses.dataclass
class FunctionTool:
    args_schema: TFunctionArgsSchema
    func: Callable[[TFunctionArgsSchema], str]
    name: Optional[str] = None


def execute_chat_model_messages(
        chat_model: BaseChatModel,
        messages: List[BaseMessage],
        chat_model_args: Optional[Dict[str, Any]] = None,
        tools: Optional[List[FunctionTool]] = None,
        spinner: Optional[Halo] = None) -> str:
    chat_model_args = chat_model_args or {}

    assert 'functions' not in chat_model_args, 'The `functions` argument is reserved for the `execute_chat_model_messages` function. If you want to add more functions use the `functions` argument to this method.'

    if tools is not None and len(tools) > 0:
        chat_model_args['functions'] = [
            pydantic_to_openai_function(function_name=tool.name, pydantic_type=tool.args_schema) for tool in tools]

    function_map = {(tool.name or tool.args_schema.__name__): tool for tool in tools or []}

    all_messages = messages.copy()

    last_message = chat_model.predict_messages(all_messages, **chat_model_args)
    function_call = last_message.additional_kwargs.get('function_call')

    while function_call is not None:
        function_name = function_call['name']
        if function_name in function_map:
            args = function_call['arguments']

            if spinner is not None:
                progress_text = f'Executing function `{function_name}`...'

                spinner.start(progress_text)

            tool = function_map[function_name]

            if tool.args_schema != str:
                args = json_string_to_pydantic(args, tool.args_schema)

            result = tool.func(args)

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


def pydantic_to_openai_function(pydantic_type: PydanticType,
                                function_name: Optional[str] = None,
                                function_description: Optional[str] = None) -> Dict:
    base_schema = pydantic_type.model_json_schema()
    del base_schema['title']
    del base_schema['description']

    description = function_description if function_description is not None else (pydantic_type.__doc__ or '')

    return {
        'name': function_name or pydantic_type.__name__,
        'description': description,
        'parameters': base_schema
    }
