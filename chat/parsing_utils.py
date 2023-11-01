from typing import Type, Optional, List

from halo import Halo
from langchain.chat_models.base import BaseChatModel

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import TOutputSchema, Chat, ChatMessage
from chat.conductors import RoundRobinChatConductor
from chat.errors import MessageCouldNotBeParsedError
from chat.participants import LangChainBasedAIChatParticipant
from chat.participants.output_parser import JSONOutputParserChatParticipant
from chat.renderers import NoChatRenderer
from chat.structured_prompt import Section, StructuredPrompt
from chat.utils import pydantic_to_json_schema


def string_output_to_pydantic(output: str,
                              chat_model: BaseChatModel,
                              output_schema: Type[TOutputSchema],
                              spinner: Optional[Halo] = None,
                              n_tries: int = 3,
                              hide_message: bool = True) -> TOutputSchema:
    return chat_messages_to_pydantic(
        chat_messages=[ChatMessage(id=1, sender_name='Unknown', text=output)],
        chat_model=chat_model,
        output_schema=output_schema,
        spinner=spinner,
        n_tries=n_tries,
        hide_message=hide_message
    )


def chat_messages_to_pydantic(chat_messages: List[ChatMessage],
                              chat_model: BaseChatModel,
                              output_schema: Type[TOutputSchema],
                              spinner: Optional[Halo] = None,
                              n_tries: int = 3,
                              hide_message: bool = True) -> TOutputSchema:
    text_to_json_ai = LangChainBasedAIChatParticipant(
        chat_model=chat_model,
        name='Chat Messages to JSON Converter',
        role='Chat Messages to JSON Converter',
        personal_mission='You will given PREVIOUS MESSAGES and a JSON SCHEMA. Your only mission is to convert the previous chat messages '
                         'to a JSON that follows the JSON SCHEMA provided. Your message should include only correct JSON.',
        other_prompt_sections=[
            Section(name='NOTES', list=[
                'Usually a JSON needs to be objective and contain no fluff. For example: "I am a human." should become {"type": "human"}',
                'However, some fields may in fact require entire sentences. For example: "I am a human." should become {"response": "I am a human."}',
            ]),
        ],
        spinner=spinner
    )
    json_parser = JSONOutputParserChatParticipant(output_schema=output_schema)

    parser_chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=NoChatRenderer(),
        initial_participants=[json_parser, text_to_json_ai],
        hide_messages=hide_message,
        max_total_messages=n_tries * 2
    )
    conductor = RoundRobinChatConductor()

    _ = conductor.initiate_chat_with_result(
        chat=parser_chat,
        initial_message=str(StructuredPrompt(
            sections=[
                Section(
                    name='Previous Messages',
                    list=[f'{m.sender_name}: {m.content}' for m in chat_messages],
                    list_item_prefix=None
                ),
                Section(
                    name='Json Schema',
                    text=str(pydantic_to_json_schema(output_schema))
                )
            ]
        ))
    )

    if json_parser.output is None:
        raise MessageCouldNotBeParsedError('An output could not be parsed from the chat messages.')

    return json_parser.output
