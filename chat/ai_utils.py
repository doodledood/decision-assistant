from typing import List, Optional, Dict, Any, Callable

from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, FunctionMessage

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
