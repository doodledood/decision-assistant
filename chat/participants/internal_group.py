from typing import Optional

from halo import Halo

from chat.base import ActiveChatParticipant, ChatConductor, Chat


class InternalGroupBasedChatParticipant(ActiveChatParticipant):
    inner_chat_conductor: ChatConductor
    inner_chat: Chat
    spinner: Optional[Halo] = None

    def __init__(self,
                 name: str,
                 role: str,
                 chat: Chat,
                 chat_conductor: ChatConductor,
                 spinner: Optional[Halo] = None,
                 **kwargs):
        super().__init__(name=name, role=role, **kwargs)

        self.inner_chat = chat
        self.inner_chat_conductor = chat_conductor
        self.spinner = spinner

    def respond_to_chat(self, chat: 'Chat') -> str:
        if self.spinner is not None:
            self.spinner.stop_and_persist(symbol='👥', text=f'{self.name}\'s group started a discussion.')
            self.spinner.start(text=f'{self.name}\'s group is discussing...')

        messages = chat.get_messages()
        conversation_str = '\n'.join([f'- {message.sender_name}: {message.content}' for message in messages])
        response = self.inner_chat_conductor.initiate_chat_with_result(
            chat=self.inner_chat,
            initial_message=f'''# ANOTHER EXTERNAL CONVERSATION\n{conversation_str}\n\n# TASK\nAs this group\'s leader, I need to respond in our group's name. What do you all think should I respond with? Let's collaborate on this.'''
        )

        if self.spinner is not None:
            self.spinner.succeed(text=f'{self.name}\'s group discussion was concluded.')

        return response
