from typing import Optional, Union, Tuple

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import ActiveChatParticipant, Chat, TOutputSchema, ChatDataBackingStore, ChatRenderer
from chat.conductors import RoundRobinChatConductor
from chat.parsing_utils import chat_messages_to_pydantic
from chat.participants import UserChatParticipant
from chat.renderers import TerminalChatRenderer


def get_response(query: str, answerer: ActiveChatParticipant,
                 backing_store: Optional[ChatDataBackingStore] = None,
                 renderer: Optional[ChatRenderer] = None) -> Tuple[Union[str, TOutputSchema], Chat]:
    user = UserChatParticipant(name='User')
    participants = [user, answerer]

    chat = Chat(
        backing_store=backing_store or InMemoryChatDataBackingStore(),
        renderer=renderer or TerminalChatRenderer(),
        initial_participants=participants,
        max_total_messages=2
    )

    chat_conductor = RoundRobinChatConductor()
    answer = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=query, from_participant=user)

    return answer, chat
