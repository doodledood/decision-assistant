import json
from typing import List, Optional, Callable, Type, TypeVar

from halo import Halo
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, FunctionMessage, HumanMessage
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel

from utils import fix_invalid_json

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
         spinner: Optional[Halo] = None,
         max_ai_messages: Optional[int] = None
         ) -> ResultSchema:
    assert len(messages) > 0, 'At least one message is required.'

    if get_user_input is None:
        get_user_input = lambda _: input('\n> ')

    if on_reply is None:
        on_reply = lambda x: None

    curr_n_ai_messages = 0
    all_messages = messages
    functions = (
            [{
                "name": '_terminate',
                "description": 'Should be called when you think you have achieved your mission or goal and ready to move on to the next step, or if asked explicitly to terminate. The result of the mission or goal should be provided as an argument. The result will be JSON-parsed, so make sure all strings in it are properly JSON formatted; especially newlines .e.g. \'\\n\' should be \'\\\\n\'. Do not use actual newlines, make sure they are properly escaped.',
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
        if spinner is not None:
            spinner.start(text='Thinking...')

        last_message = chat_model.predict_messages(all_messages, functions=functions)

        if spinner is not None:
            if last_message.content != '':
                spinner.stop_and_persist(symbol='🤖', text=last_message.content)
            else:
                spinner.stop()

        all_messages.append(last_message)

        function_call = last_message.additional_kwargs.get('function_call')
        if function_call is not None:
            if function_call['name'] == '_terminate':
                args = json.loads(fix_invalid_json(function_call['arguments']))

                result = args['result']
                if result_schema is not None:
                    if isinstance(result, str):
                        result = json_string_to_pydantic(str(result), result_schema)
                    else:
                        result = result_schema.parse_obj(result)

                return result

            for tool in tools:
                if tool.name == function_call['name']:
                    args = function_call['arguments']

                    if spinner is not None:
                        if hasattr(tool, 'progress_text'):
                            progress_text = tool.progress_text
                        else:
                            progress_text = 'Executing function call...'

                        spinner.start(progress_text)

                    result = tool.run(args)
                    messages.append(FunctionMessage(
                        name=tool.name,
                        content=result or 'None'
                    ))

                    break
        else:
            last_message = messages[-1]
            on_reply(last_message.content)

            curr_n_ai_messages += 1

            if max_ai_messages is not None and curr_n_ai_messages >= max_ai_messages:
                if curr_n_ai_messages >= max_ai_messages + 1:
                    raise Exception(
                        f'AI did not terminate when asked to do so. This is the last message: `{last_message}`')

                user_input = 'Please now "_terminate" immediately with the result of your mission.'
            else:
                user_input = get_user_input(messages)

            all_messages.append(HumanMessage(content=user_input))
