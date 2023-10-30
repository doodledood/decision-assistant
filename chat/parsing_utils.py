from typing import Type, Optional

from halo import Halo
from langchain.chat_models.base import BaseChatModel

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import TOutputSchema, Chat
from chat.conductors import RoundRobinChatConductor
from chat.errors import MessageCouldNotBeParsedError
from chat.participants import LangChainBasedAIChatParticipant
from chat.participants.output_parser import JSONOutputParserChatParticipant
from chat.renderers import NoChatRenderer
from chat.utils import pydantic_to_json_schema


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
        backing_store=InMemoryChatDataBackingStore(),
        renderer=NoChatRenderer(),
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
