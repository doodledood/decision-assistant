import json
from typing import Optional, List, Callable, TypeVar, Type

from dotenv import load_dotenv
from fire import Fire
from halo import Halo
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, FunctionMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from pydantic.v1 import BaseModel, Field

import system_prompts

ResultSchema = TypeVar("T", bound=BaseModel)


def chat(chat_model: ChatOpenAI,
         messages: List[BaseMessage],
         tools: Optional[List[Tool]] = None,
         get_user_input: Optional[Callable[[List[BaseMessage]], str]] = None,
         on_reply: Optional[Callable[[str], None]] = None,
         result_schema: Optional[Type[BaseModel]] = None
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
                        } if result_schema is None else result_schema.model_json_schema(),
                    },
                    "required": ["result"],
                    "type": "object"
                }
            }] +
            [format_tool_to_openai_function(tool) for tool in (tools or [])]
    )

    while True:
        last_message = chat_model.predict_messages(all_messages, functions=functions)
        all_messages.append(last_message)

        function_call = last_message.additional_kwargs.get('function_call')
        if function_call is not None:
            if function_call['name'] == '_terminate':
                args = json.loads(function_call['arguments'])

                result = args['result']
                if result_schema is not None:
                    result = result_schema.model_validate_json(result)

                return result

            for tool in tools:
                if tool.name == function_call['name']:
                    orig_args = args = function_call['arguments']
                    progress_text = 'Executing function call...'

                    if tool.args_schema is not None:
                        try:
                            args = tool.args_schema.model_validate_json(args)
                        except AttributeError:
                            args = tool.args_schema.parse_raw(args)

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


def run_decision_assistant(goal: Optional[str] = None, llm_temperature: float = 0.0, llm_model='gpt-4-0613'):
    chat_model = ChatOpenAI(temperature=llm_temperature, model=llm_model, streaming=True,
                            callbacks=[StreamingStdOutCallbackHandler()])

    if goal is None:
        goal = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.goal_identification_system_prompt),
            HumanMessage(content="Hey")
        ])

    print(goal)


if __name__ == '__main__':
    load_dotenv()

    Fire(run_decision_assistant)
