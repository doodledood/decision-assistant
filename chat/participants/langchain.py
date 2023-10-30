import dataclasses
from typing import Dict, Any, List, Callable, Optional, Tuple

from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage

from chat.ai_utils import execute_chat_model_messages, FunctionTool
from chat.base import ChatMessage, Chat, ActiveChatParticipant
from chat.structured_prompt import Section, StructuredPrompt



class LangChainBasedAIChatParticipant(ActiveChatParticipant):
    personal_mission: str
    chat_model: BaseChatModel
    chat_model_args: Dict[str, Any]
    other_prompt_sections: List[Section]
    tools: Optional[List[FunctionTool]] = None,
    spinner: Optional[Halo] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 name: str,
                 chat_model: BaseChatModel,
                 symbol: str = '🤖',
                 role: str = 'AI Assistant',
                 personal_mission: str = 'Be a helpful AI assistant.',
                 other_prompt_sections: Optional[List[Section]] = None,
                 tools: Optional[List[FunctionTool]] = None,
                 chat_model_args: Optional[Dict[str, Any]] = None,
                 spinner: Optional[Halo] = None,
                 **kwargs
                 ):
        super().__init__(name=name, symbol=symbol, role=role, **kwargs)

        self.chat_model = chat_model
        self.chat_model_args = chat_model_args or {}
        self.other_prompt_sections = other_prompt_sections or []
        self.tools = tools
        self.spinner = spinner
        self.personal_mission = personal_mission

    def create_system_message(self, chat: 'Chat'):
        active_participants = chat.get_active_participants()
        system_message = StructuredPrompt(
            sections=[
                Section(name='Personal Mission', text=self.personal_mission),
                Section(name='Name', text=self.name),
                Section(name='Role', text=self.role),
                Section(name='Chat', sub_sections=[
                    Section(name='Goal', text=chat.goal or 'No explicit chat goal provided.'),
                    Section(name='Participants', text='\n'.join(
                        [f'- Name: "{p.name}", Role: "{p.role}"{" -> This is you." if p.name == self.name else ""}' \
                         for p in active_participants])),
                    Section(name='Rules', list=[
                        'You do not have to respond directly to the one who sent you a message. You can respond to anyone in the group chat.',
                        'You cannot have private conversations with other participants. Everyone can see all messages sent by all other participants.',
                    ]),
                    Section(name='Messages', list=[
                        'Include all participants messages, including your own',
                        'They are prefixed by the sender\'s name (could also be everyone). For context only; it\'s not actually part of the message they sent. Example: "John: Hello, how are you?"',
                        'Some messages could have been sent by participants who are no longer a part of this conversation. Use their contents for context only; do not talk to them.',
                    ]),
                    Section(name='Well-Formatted Response Examples', list=[
                        '"Hello, how are you?"',
                        '"I am doing well, thanks. How are you?"',
                    ]),
                    Section(name='Badly-Formatted Response Examples', list=[
                        '"John: Hello, how are you?"',
                        '"Assistant: I am doing well, thanks. How are you?"',
                    ]),
                ]),
                *self.other_prompt_sections,
            ]
        )
        return str(system_message)

    def chat_messages_to_chat_model_messages(self, chat_messages: List[ChatMessage]) -> List[BaseMessage]:
        messages = []
        for message in chat_messages:
            content = \
                f'{message.sender_name}: {message.content}'
            if message.sender_name == self.name:
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        if len(messages) == 0:
            messages.append(HumanMessage(content=f'SYSTEM: The chat has started.'))

        return messages

    def respond_to_chat(self, chat: Chat) -> str:
        if self.spinner is not None:
            self.spinner.start(text=f'{self.name} ({self.role}) is thinking...')

        system_message = self.create_system_message(chat=chat)

        chat_messages = chat.get_messages()

        all_messages = self.chat_messages_to_chat_model_messages(chat_messages)
        all_messages = [
            SystemMessage(content=system_message),
            *all_messages
        ]

        message_content = self.execute_messages(messages=all_messages)

        if self.spinner is not None:
            self.spinner.stop()

        potential_prefix = f'{self.name}:'
        if message_content.startswith(potential_prefix):
            message_content = message_content[len(potential_prefix):].strip()

        return message_content

    def execute_messages(self, messages: List[BaseMessage]) -> str:
        return execute_chat_model_messages(
            messages=messages,
            chat_model=self.chat_model,
            tools=self.tools,
            spinner=self.spinner,
            chat_model_args=self.chat_model_args
        )
