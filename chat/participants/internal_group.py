from typing import Optional

from halo import Halo

from chat.base import ActiveChatParticipant, ChatConductor, Chat
from chat.structured_string import StructuredString, Section
from chat.use_cases.request_response import get_response


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
        # Make sure the inner chat is empty
        self.inner_chat.clear_messages()

        prev_spinner_text = None
        if self.spinner is not None:
            prev_spinner_text = self.spinner.text
            self.spinner.stop_and_persist(symbol='👥', text=f'{self.name}\'s group started a discussion.')
            self.spinner.start(text=f'{self.name}\'s group is discussing...')

        messages = chat.get_messages()
        conversation_str = '\n'.join([f'- {message.sender_name}: {message.content}' for message in messages])

        leader = self.inner_chat.get_active_participants()[0]

        request_for_group, _ = get_response(
            query='Please translate the request for yourself in the external conversation into a succinct request for '
                  'your internal group (as their team leader). This is the external conversation:'
                  f'\n```{conversation_str}```\n\nThe group should understand exactly what to discuss and what to '
                  'decide on based on this.',
            answerer=leader)

        group_response = self.inner_chat_conductor.initiate_chat_with_result(
            chat=self.inner_chat,
            initial_message=request_for_group
        )

        if self.spinner is not None:
            self.spinner.succeed(text=f'{self.name}\'s group discussion was concluded.')
            if prev_spinner_text is not None:
                self.spinner.start(text=prev_spinner_text)

        messages = self.inner_chat.get_messages()
        group_response_conversation_str = '\n'.join(
            [f'- {message.sender_name}: {message.content}' for message in messages])

        leader_response_back, _ = get_response(
            query=str(StructuredString(sections=[
                Section(name='External Conversation',
                        text=conversation_str),
                Section(name='Internal Group Conversation',
                        text=group_response_conversation_str),
                Section(name='Task',
                        text='You are a part of the original conversation. You (the group\'s leader) and your group '
                             'have collaborated on a response for the EXTERNAL CONVERSATION. Please transform the '
                             'INTERNAL GROUP CONVERSATION into a PROPER response for this conversation. '
                             'The response you give will be the response you will give to the external conversation, '
                             'verbatim: it has to fit the context of the external conversation.')
            ])), answerer=leader)

        return leader_response_back
