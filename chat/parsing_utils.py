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
from chat.structured_string import Section
from chat.utils import pydantic_to_json_schema


def string_output_to_pydantic(output: str,
                              chat_model: BaseChatModel,
                              output_schema: Type[TOutputSchema],
                              spinner: Optional[Halo] = None,
                              n_tries: int = 3,
                              hide_message: bool = True) -> TOutputSchema:
    return chat_messages_to_pydantic(
        chat_messages=[ChatMessage(id=1, sender_name='Unknown', content=output)],
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
        personal_mission='Your only purpose is to convert the previous chat messages (usually the last one)'
                         'to a valid and logical JSON that follows the JSON SCHEMA provided. Your message should '
                         'include only correct JSON. No fluff.',
        other_prompt_sections=[
            Section(name='JSON SCHEMA', text=str(pydantic_to_json_schema(output_schema)))
        ],
        ignore_group_chat_environment=True,
        spinner=spinner
    )
    json_parser = JSONOutputParserChatParticipant(output_schema=output_schema)

    # Remove TERMINATE if present so the chat conductor doesn't end the chat prematurely
    if len(chat_messages) > 0:
        chat_messages = chat_messages.copy()
        last_message = chat_messages[-1]

        try:
            # Chop the content at the last instance of the word TERMINATE in the content
            idx = last_message.content.rindex('TERMINATE')
            new_content = last_message.content[:idx].strip()

            last_message = ChatMessage(
                id=last_message.id,
                sender_name=last_message.sender_name,
                content=new_content
            )

            chat_messages[-1] = last_message
        except ValueError:
            pass

    parser_chat = Chat(
        goal='Convert the chat contents to a valid and logical JSON.',
        backing_store=InMemoryChatDataBackingStore(messages=chat_messages),
        renderer=NoChatRenderer(),
        initial_participants=[text_to_json_ai, json_parser],
        hide_messages=hide_message,
        max_total_messages=1 + (n_tries - 1) * 2
    )
    conductor = RoundRobinChatConductor()

    _ = conductor.initiate_chat_with_result(chat=parser_chat)

    if json_parser.output is None:
        raise MessageCouldNotBeParsedError('An output could not be parsed from the chat messages.')

    return json_parser.output
