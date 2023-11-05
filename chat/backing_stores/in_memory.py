from typing import List, Dict, Optional

from chat.base import ChatDataBackingStore, ChatMessage, ChatParticipant, ActiveChatParticipant
from chat.errors import ChatParticipantAlreadyJoinedToChatError, ChatParticipantNotJoinedToChatError


class InMemoryChatDataBackingStore(ChatDataBackingStore):
    messages: List[ChatMessage]
    participants: Dict[str, ChatParticipant]
    last_message_id: Optional[int] = None

    def __init__(self, messages: Optional[List[ChatMessage]] = None,
                 participants: Optional[List[ChatParticipant]] = None):
        self.messages = messages or []
        self.participants = {participant.name: participant for participant in (participants or [])}
        self.last_message_id = None if len(self.messages) == 0 else self.messages[-1].id

    def get_messages(self) -> List[ChatMessage]:
        return self.messages

    def add_message(self, sender_name: str, content: str) -> ChatMessage:
        self.last_message_id = self.last_message_id + 1 if self.last_message_id is not None else 1

        message = ChatMessage(
            id=self.last_message_id,
            sender_name=sender_name,
            content=content
        )

        self.messages.append(message)

        return message

    def get_active_participants(self) -> List[ActiveChatParticipant]:
        participants = list(self.participants.values())
        participants = [participant for participant in participants if isinstance(participant, ActiveChatParticipant)]

        return participants

    def get_non_active_participants(self) -> List[ChatParticipant]:
        participants = list(self.participants.values())
        participants = [participant for participant in participants if
                        not isinstance(participant, ActiveChatParticipant)]

        return participants

    def get_active_participant_by_name(self, name: str) -> Optional[ActiveChatParticipant]:
        if name not in self.participants:
            return None

        participant = self.participants[name]
        if not isinstance(participant, ActiveChatParticipant):
            return None

        return participant

    def get_non_active_participant_by_name(self, name: str) -> Optional[ChatParticipant]:
        if name not in self.participants:
            return None

        participant = self.participants[name]
        if isinstance(participant, ActiveChatParticipant):
            return None

        return participant

    def add_participant(self, participant: ChatParticipant):
        if participant.name in self.participants:
            raise ChatParticipantAlreadyJoinedToChatError(participant.name)

        self.participants[participant.name] = participant

    def remove_participant(self, participant: ChatParticipant):
        if participant.name not in self.participants:
            raise ChatParticipantNotJoinedToChatError(participant.name)

        self.participants.pop(participant.name)

    def has_active_participant_with_name(self, participant_name: str) -> bool:
        return participant_name in self.participants
