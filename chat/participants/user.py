from chat.base import ActiveChatParticipant, Chat


class UserChatParticipant(ActiveChatParticipant):
    def __init__(self, name: str = 'User', **kwargs):
        super().__init__(name, role='User', messages_hidden=True, **kwargs)

    def respond_to_chat(self, chat: Chat) -> str:
        return input(f'👤 ({self.name}): ')
