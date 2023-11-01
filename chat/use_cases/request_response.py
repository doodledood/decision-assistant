from typing import Optional, Union

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import ActiveChatParticipant, Chat, TOutputSchema
from chat.conductors import RoundRobinChatConductor
from chat.parsing_utils import chat_messages_to_pydantic
from chat.participants import UserChatParticipant
from chat.renderers import TerminalChatRenderer


def get_answer(query: str, answerer: ActiveChatParticipant,
               output_schema: Optional[TOutputSchema] = None) -> Union[str, TOutputSchema]:
    user = UserChatParticipant(name='User')
    participants = [user, answerer]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants,
        max_total_messages=2
    )

    chat_conductor = RoundRobinChatConductor()
    answer = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=query, from_participant=user)

    if output_schema is not None:
        answer = chat_messages_to_pydantic(chat.get_messages(), output_schema)

    return answer
