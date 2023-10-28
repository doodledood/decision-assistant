import abc
import json
from json import JSONDecodeError
from typing import List, Optional, Callable, Type, TypeVar, Dict, Any

from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, FunctionMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel
from pydantic.v1 import ValidationError

from chat.ai_utils import execute_chat_model_messages, pydantic_to_openai_function
from chat.errors import NotEnoughActiveParticipantsInChatError, ChatParticipantNotJoinedToChatError, \
    ChatParticipantAlreadyJoinedToChatError, NoMessagesInChatError, MessageCouldNotBeParsedError
from chat.structured_prompt import StructuredPrompt, Section
from utils import fix_invalid_json

TOutputSchema = TypeVar("TOutputSchema", bound=BaseModel)


def pydantic_to_json_schema(pydantic_model: Type[BaseModel]) -> dict:
    try:
        return pydantic_model.model_json_schema()
    except AttributeError:
        return pydantic_model.schema()


def json_string_to_pydantic(json_string: str, pydantic_model: Type[BaseModel]) -> BaseModel:
    try:
        return pydantic_model.model_validate_json(json_string)
    except AttributeError:
        return pydantic_model.parse_raw(json_string)


terminate_now_message_content = 'Please now "_terminate" immediately with the result of your mission.'


def chat(chat_model: BaseChatModel,
         messages: List[BaseMessage],
         tools: Optional[List[Tool]] = None,
         get_user_input: Optional[Callable[[List[BaseMessage]], str]] = None,
         on_reply: Optional[Callable[[str], None]] = None,
         result_schema: Optional[Type[BaseModel]] = None,
         spinner: Optional[Halo] = None,
         max_ai_messages: Optional[int] = None,
         max_consecutive_arg_error_count: int = 3,
         get_immediate_answer: bool = False,
         ) -> TOutputSchema:
    assert len(messages) > 0, 'At least one message is required.'

    if get_user_input is None:
        get_user_input = lambda _: input('\n👤: ')

    if on_reply is None:
        on_reply = lambda message: print(f'\n🤖: {message}')

    if get_immediate_answer:
        get_user_input = lambda _: terminate_now_message_content
        on_reply = lambda _: None
        max_ai_messages = 1

    curr_n_ai_messages = 0
    all_messages = messages
    functions = (
            [{
                "name": '_terminate',
                "description": 'Should be called when you think you have achieved your mission or goal and ready to move on to the next step, or if asked explicitly to terminate. The result of the mission or goal should be provided as an argument. The result will be JSON-parsed, so make sure all strings in it are properly JSON formatted; especially newlines .e.g. \'\\n\' should be \'\\\\n\'. Do not use actual newlines, make sure they are properly escaped.',
                "parameters": {
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "The result of the mission or goal."
                        } if result_schema is None else pydantic_to_json_schema(result_schema),
                    },
                    "required": ["result"],
                    "type": "object"
                }
            }] +
            [format_tool_to_openai_function(tool) for tool in (tools or [])]
    )

    consecutive_arg_error_count = 0

    while True:
        if spinner is not None:
            spinner.start(text='Thinking...')

        last_message = chat_model.predict_messages(all_messages, functions=functions)

        if spinner is not None:
            spinner.stop()

        all_messages.append(last_message)

        function_call = last_message.additional_kwargs.get('function_call')
        if function_call is not None:
            if function_call['name'] == '_terminate':
                try:
                    args = json.loads(fix_invalid_json(function_call['arguments']))

                    result = args['result']
                    if result_schema is not None:
                        if isinstance(result, str):
                            result = json_string_to_pydantic(str(result), result_schema)
                        else:
                            result = result_schema.parse_obj(result)

                    return result
                except Exception as e:
                    if type(e) not in (JSONDecodeError, ValidationError):
                        raise

                    if consecutive_arg_error_count >= max_consecutive_arg_error_count - 1:
                        raise

                    all_messages.append(FunctionMessage(
                        name=function_call['name'],
                        content=f'ERROR: Arguments to the function call were not valid JSON or did not follow the given schema for the function argss. Please try again. Error: {e}'
                    ))
                    consecutive_arg_error_count += 1
            else:
                for tool in tools:
                    if tool.name == function_call['name']:
                        args = function_call['arguments']

                        if spinner is not None:
                            if hasattr(tool, 'progress_text'):
                                progress_text = tool.progress_text
                            else:
                                progress_text = 'Executing function call...'

                            spinner.start(progress_text)

                        result = tool.run(args)
                        all_messages.append(FunctionMessage(
                            name=tool.name,
                            content=result or 'None'
                        ))

                        break
        else:
            on_reply(last_message.content)

            curr_n_ai_messages += 1

            if max_ai_messages is not None and curr_n_ai_messages >= max_ai_messages:
                if curr_n_ai_messages >= max_ai_messages + 1:
                    raise Exception(
                        f'AI did not terminate when asked to do so. This is the last message: `{last_message}`')

                user_input = terminate_now_message_content
                max_ai_messages = curr_n_ai_messages
            else:
                user_input = get_user_input(all_messages)

            all_messages.append(HumanMessage(content=user_input))


class ChatParticipant(abc.ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    def on_new_chat_message(self, chat: 'ChatRoom', message: 'ChatMessage'):
        pass

    def on_chat_ended(self, chat: 'ChatRoom'):
        pass

    def on_participant_joined_chat(self, chat: 'ChatRoom', participant: 'ChatParticipant'):
        pass

    def on_participant_left_chat(self, chat: 'ChatRoom', participant: 'ChatParticipant'):
        pass

    def on_new_chat_leader_elected(self, chat: 'ChatRoom', new_leader: Optional['ActiveChatParticipant']):
        pass


class ActiveChatParticipant(ChatParticipant):
    name: str
    symbol: str
    role: str
    messages_hidden: bool

    def __init__(self, name: str, symbol: str = '👤', role: str = 'Chat Participant', messages_hidden: bool = False):
        super().__init__(name=name)

        self.symbol = symbol
        self.role = role
        self.messages_hidden = messages_hidden

    @abc.abstractmethod
    def respond_to_chat(self, chat: 'ChatRoom') -> str:
        raise NotImplementedError()


class ChatMessage(BaseModel):
    id: int
    sender_name: str
    content: str


class ChatConductor(abc.ABC):
    @abc.abstractmethod
    def select_next_speaker(self, chat: 'ChatRoom') -> Optional[ActiveChatParticipant]:
        raise NotImplementedError()

    def get_chat_result(self, chat: 'ChatRoom') -> str:
        messages = chat.chat_backing_store.get_messages()
        if len(messages) == 0:
            return ''

        last_message = messages[-1]

        return last_message.content

    def elect_new_chat_leader(self, chat: 'ChatRoom') -> ActiveChatParticipant:
        participants = chat.chat_backing_store.get_active_participants()
        if len(participants) == 0:
            raise NotEnoughActiveParticipantsInChatError()

        return next(iter(participants))


class RoundRobinChatConductor(ChatConductor):
    def select_next_speaker(self, chat: 'ChatRoom') -> Optional[ActiveChatParticipant]:
        active_participants = chat.chat_backing_store.get_active_participants()
        if len(active_participants) <= 1:
            return None

        messages = chat.chat_backing_store.get_messages()
        last_message = messages[-1] if len(messages) > 0 else None

        if last_message is not None and self.is_termination_message(last_message):
            return None

        last_speaker = last_message.sender_name if last_message is not None else None
        if last_speaker is None:
            return next(iter(active_participants))

        # Rotate to the next participant in the list.
        participant_names = [participant.name for participant in active_participants]
        last_speaker_index = participant_names.index(last_speaker)
        next_speaker_index = (last_speaker_index + 1) % len(participant_names)
        next_speaker_name = participant_names[next_speaker_index]
        next_speaker = chat.chat_backing_store.get_active_participant_by_name(next_speaker_name)
        if next_speaker is None or not isinstance(next_speaker, ActiveChatParticipant):
            raise ChatParticipantNotJoinedToChatError(next_speaker_name)

        return next_speaker

    def get_chat_result(self, chat: 'ChatRoom') -> str:
        result = super().get_chat_result(chat=chat)
        result = result.replace('TERMINATE', '').strip()

        return result

    def is_termination_message(self, message: ChatMessage):
        return message.content.strip().endswith('TERMINATE')


class LangChainBasedAIChatConductor(ChatConductor):
    chat_model: BaseChatModel
    chat_model_args: Dict[str, Any]
    functions: Dict[str, Callable[[Any], str]]
    speaker_interaction_schema: Optional[str]
    termination_condition: str = f'''Terminate the chat on the following conditions:
    - When you think it has concluded
    - If one of the participants asks you to terminate it or has finished their sentence with "TERMINATE".'''
    spinner: Optional[Halo] = None

    def __init__(self,
                 chat_model: BaseChatModel,
                 speaker_interaction_schema: Optional[str] = None,
                 termination_condition: Optional[str] = None,
                 spinner: Optional[Halo] = None,
                 functions: Optional[Dict[str, Callable[[Any], str]]] = None,
                 chat_model_args: Optional[Dict[str, Any]] = None):
        self.chat_model = chat_model
        self.chat_model_args = chat_model_args or {}
        self.functions = functions or {}
        self.speaker_interaction_schema = speaker_interaction_schema
        self.termination_condition = termination_condition
        self.spinner = spinner

    def create_system_prompt(self, chat: 'ChatRoom') -> str:
        participants = chat.chat_backing_store.get_active_participants()
        system_message = StructuredPrompt(sections=[
            Section(name='Mission',
                    text='Select the next speaker in the conversation based on the previous messages in the conversation and an optional SPEAKER INTERACTION SCHEMA. If it seems to you that the chat should end instead of selecting a next speaker, terminate it.'),
            Section(name='Participants',
                    list=[f'{participant.name} ({participant.role})' for participant in participants]),
            Section(name='Rules', list=[
                'You can only select one of the participants in the group chat.'
            ]),
            Section(name='Speaker Interaction Schema',
                    text=self.speaker_interaction_schema or 'Not provided. Use your best judgement.'),
            Section(name='Termination Condition', text=self.termination_condition),
            Section(name='Process', list=[
                'Look at the last message in the conversation and determine who should speak next based on the SPEAKER INTERACTION SCHEMA, if provided.',
                'If based on TERMINATION CONDITION you determine that the chat should end, you should return the string TERMINATE instead of a participant name.'
            ]),
            Section(name='Input', text='The previous messages in the conversation.'),
            Section(name='Output',
                    text='The name of the next speaker in the conversation. Or, TERMINATE if the chat should end, instead.'),
            Section(name='Example Outputs', list=[
                '"John"',
                '"TERMINATE"'
            ])
        ])

        return str(system_message)

    def create_first_human_prompt(self, chat: 'ChatRoom') -> str:
        messages = chat.chat_backing_store.get_messages()
        messages_list = [f'- {message.sender_name}: {message.content}' for message in messages]

        prompt = StructuredPrompt(sections=[
            Section(name='Previous Messages',
                    text='No messages yet.' if len(messages_list) == 0 else None,
                    list=messages_list if len(messages_list) > 0 else []
                    ),
        ])

        return str(prompt)

    def select_next_speaker(self, chat: 'ChatRoom') -> Optional[ActiveChatParticipant]:
        participants = chat.chat_backing_store.get_active_participants()
        if len(participants) <= 1:
            return None

        if self.spinner is not None:
            self.spinner.start(text='AI Chat Conductor is selecting the next speaker...')

        # Ask the AI to select the next speaker.
        messages = [
            SystemMessage(content=self.create_system_prompt(chat=chat)),
            HumanMessage(content=self.create_first_human_prompt(chat=chat))
        ]

        result = self.execute_messages(messages=messages)
        next_speaker_name = result.strip()

        while not chat.chat_backing_store.has_active_participant_with_name(
                next_speaker_name) and next_speaker_name != 'TERMINATE':
            messages.append(AIMessage(content=next_speaker_name))
            messages.append(HumanMessage(
                content=f'Speaker "{next_speaker_name}" is not a participant in the chat. Choose another one.'))

            result = self.execute_messages(messages=messages)
            next_speaker_name = result.strip()

        if next_speaker_name == 'TERMINATE':
            if self.spinner is not None:
                self.spinner.stop_and_persist(symbol='👥', text='The Chat Conductor has decided to terminate the chat.')

            return None

        next_speaker = chat.chat_backing_store.get_active_participant_by_name(next_speaker_name)
        if next_speaker is None:
            raise ChatParticipantNotJoinedToChatError(next_speaker_name)

        if self.spinner is not None:
            self.spinner.succeed(text=f'The Chat Conductor has selected "{next_speaker_name}" as the next speaker.')

        return next_speaker

    def execute_messages(self, messages: List[BaseMessage]) -> str:
        return execute_chat_model_messages(
            messages=messages,
            chat_model=self.chat_model,
            functions=self.functions,
            spinner=self.spinner,
            chat_model_args=self.chat_model_args
        )


class ChatRenderer(abc.ABC):
    def render_new_chat_message(self, chat: 'ChatRoom', message: ChatMessage):
        raise NotImplementedError()


class NoChatRenderer(ChatRenderer):
    def render_new_chat_message(self, chat: 'ChatRoom', message: ChatMessage):
        pass


class TerminalChatRenderer(ChatRenderer):
    def render_new_chat_message(self, chat: 'ChatRoom', message: ChatMessage):
        if chat.hide_messages:
            return

        sender = chat.chat_backing_store.get_active_participant_by_name(message.sender_name)
        if sender is None:
            symbol = '❓'
        else:
            if sender.messages_hidden:
                return

            symbol = sender.symbol

        print(f'{symbol} {message.sender_name}: {message.content}')


class ChatDataBackingStore(abc.ABC):
    @abc.abstractmethod
    def get_messages(self) -> List[ChatMessage]:
        raise NotImplementedError()

    @abc.abstractmethod
    def add_message(self, sender_name: str, content: str) -> ChatMessage:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_active_participants(self) -> List[ActiveChatParticipant]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_non_active_participants(self) -> List[ChatParticipant]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_active_participant_by_name(self, name: str) -> Optional[ActiveChatParticipant]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_non_active_participant_by_name(self, name: str) -> Optional[ChatParticipant]:
        raise NotImplementedError()

    @abc.abstractmethod
    def add_participant(self, participant: ChatParticipant):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_participant(self, participant: ChatParticipant):
        raise NotImplementedError()

    @abc.abstractmethod
    def has_active_participant_with_name(self, participant_name: str) -> bool:
        raise NotImplementedError()


class InMemoryChatDataBackingStore(ChatDataBackingStore):
    messages: List[ChatMessage]
    participants: Dict[str, ChatParticipant]
    last_message_id: Optional[int] = None

    def __init__(self, messages: Optional[List[ChatMessage]] = None,
                 participants: Optional[List[ChatParticipant]] = None):
        self.messages = sorted(messages or [], key=lambda m: m.id)
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


class ChatRoom:
    chat_backing_store: ChatDataBackingStore
    chat_conductor: ChatConductor
    chat_renderer: ChatRenderer
    description: str
    chat_leader: Optional[ActiveChatParticipant] = None
    max_total_messages: Optional[int] = None
    hide_messages: bool = False

    def __init__(
            self,
            initial_participants: Optional[List[ChatParticipant]] = None,
            chat_backing_data_store: Optional[ChatDataBackingStore] = None,
            chat_conductor: Optional[ChatConductor] = None,
            chat_renderer: Optional[ChatRenderer] = None,
            max_total_messages: Optional[int] = None,
            hide_messages: bool = False,
            description: str = 'This chat room is a regular group chat. Everybody can talk to everybody else.'
    ):
        assert max_total_messages is None or max_total_messages > 0, 'Max total messages must be None or greater than 0.'

        self.chat_backing_store = chat_backing_data_store or InMemoryChatDataBackingStore()
        self.chat_conductor = chat_conductor or RoundRobinChatConductor()
        self.chat_renderer = chat_renderer or TerminalChatRenderer()
        self.hide_messages = hide_messages
        self.max_total_messages = max_total_messages
        self.description = description

        for i, participant in enumerate(initial_participants):
            self.add_participant(participant,
                                 should_elect_new_leader=self.chat_leader is None and isinstance(participant,
                                                                                                 ActiveChatParticipant))

    def add_participant(self, participant: ChatParticipant, should_elect_new_leader: bool = False):
        self.chat_backing_store.add_participant(participant)

        all_participants = self.chat_backing_store.get_active_participants() + self.chat_backing_store.get_non_active_participants()

        if self.chat_leader is None and isinstance(participant, ActiveChatParticipant):
            self.chat_leader = participant
        elif should_elect_new_leader:
            self.chat_leader = self.chat_conductor.elect_new_chat_leader(chat=self)

            for participant in all_participants:
                participant.on_new_chat_leader_elected(chat=self, new_leader=self.chat_leader)

        for participant in all_participants:
            participant.on_participant_joined_chat(chat=self, participant=participant)

    def remove_participant(self, participant: ChatParticipant):
        self.chat_backing_store.remove_participant(participant)

        active_participants = self.chat_backing_store.get_active_participants()
        non_active_participants = self.chat_backing_store.get_non_active_participants()
        all_participants = active_participants + non_active_participants

        for participant in all_participants:
            participant.on_participant_left_chat(chat=self, participant=participant)

        if self.chat_leader == participant:
            if len(active_participants) == 0:
                self.chat_leader = None
            else:
                self.chat_leader = self.chat_conductor.elect_new_chat_leader(chat=self)

            for participant in all_participants:
                participant.on_new_chat_leader_elected(chat=self, new_leader=self.chat_leader)

    def receive_message(self, sender_name: str, content: str):
        sender = self.chat_backing_store.get_active_participant_by_name(sender_name)
        if sender is None:
            raise ChatParticipantNotJoinedToChatError(sender_name)

        message = self.chat_backing_store.add_message(sender_name=sender_name, content=content)

        self.chat_renderer.render_new_chat_message(chat=self, message=message)

        active_participants = self.chat_backing_store.get_active_participants()
        non_active_participants = self.chat_backing_store.get_non_active_participants()
        all_participants = active_participants + non_active_participants

        for participant in all_participants:
            participant.on_new_chat_message(chat=self, message=message)

    def initiate_chat_with_result(
            self,
            initial_message: Optional[str] = None
    ) -> str:
        active_participants = self.chat_backing_store.get_active_participants()
        if len(active_participants) <= 1:
            raise NotEnoughActiveParticipantsInChatError(len(participants))

        if initial_message is not None:
            self.receive_message(sender_name=self.chat_leader.name, content=initial_message)

        next_speaker = self.chat_conductor.select_next_speaker(chat=self)
        while next_speaker is not None:
            messages = self.chat_backing_store.get_messages()
            if self.max_total_messages is not None and len(messages) >= self.max_total_messages:
                break

            message_content = next_speaker.respond_to_chat(chat=self)

            self.receive_message(sender_name=next_speaker.name, content=message_content)

            next_speaker = self.chat_conductor.select_next_speaker(chat=self)

        active_participants = self.chat_backing_store.get_active_participants()
        non_active_participants = self.chat_backing_store.get_non_active_participants()
        all_participants = active_participants + non_active_participants

        for participant in all_participants:
            participant.on_chat_ended(chat=self)

        return self.chat_conductor.get_chat_result(chat=self)


class UserChatParticipant(ActiveChatParticipant):
    def __init__(self, name: str = 'User', **kwargs):
        super().__init__(name, role='User', messages_hidden=True, **kwargs)

    def respond_to_chat(self, chat: 'ChatRoom') -> str:
        return input(f'👤 ({self.name}): ')


class LangChainBasedAIChatParticipant(ActiveChatParticipant):
    mission: str
    chat_model: BaseChatModel
    chat_model_args: Dict[str, Any]
    other_prompt_sections: List[Section]
    functions: Dict[str, Callable[[Any], str]]
    spinner: Optional[Halo] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 name: str,
                 chat_model: BaseChatModel,
                 symbol: str = '🤖',
                 role: str = 'AI Assistant',
                 mission: str = 'Be a helpful AI assistant.',
                 other_prompt_sections: Optional[List[Section]] = None,
                 functions: Optional[Dict[str, Callable[[Any], str]]] = None,
                 chat_model_args: Optional[Dict[str, Any]] = None,
                 spinner: Optional[Halo] = None,
                 **kwargs
                 ):
        super().__init__(name=name, symbol=symbol, role=role, **kwargs)

        self.chat_model = chat_model
        self.chat_model_args = chat_model_args or {}
        self.other_prompt_sections = other_prompt_sections or []
        self.functions = functions or {}
        self.spinner = spinner
        self.mission = mission

    def create_system_message(self, chat: 'ChatRoom'):
        active_participants = chat.chat_backing_store.get_active_participants()
        system_message = StructuredPrompt(
            sections=[
                Section(name='Mission', text=self.mission),
                Section(name='Name', text=self.name),
                Section(name='Role', text=self.role),
                Section(name='Chat Room', text=chat.description, sub_sections=[
                    Section(name='Description', text=chat.description),
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
                    *self.other_prompt_sections,
                    Section(name='Good Responses Examples', list=[
                        '"Hello, how are you?"',
                        '"I am doing well, thanks. How are you?"',
                    ]),
                    Section(name='Bad Responses Examples', list=[
                        '"John: Hello, how are you?"',
                        '"Assistant: I am doing well, thanks. How are you?"',
                    ]),
                ])
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

    def respond_to_chat(self, chat: 'ChatRoom') -> str:
        if self.spinner is not None:
            self.spinner.start(text=f'{self.name} ({self.role}) is thinking...')

        system_message = self.create_system_message(chat=chat)

        chat_messages = chat.chat_backing_store.get_messages()

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
            functions=self.functions,
            spinner=self.spinner,
            chat_model_args=self.chat_model_args
        )


class GroupBasedChatParticipant(ActiveChatParticipant):
    inner_chat: ChatRoom
    spinner: Optional[Halo] = None

    def __init__(self,
                 chat: ChatRoom,
                 spinner: Optional[Halo] = None,
                 **kwargs):
        if chat.chat_leader is None:
            raise NotEnoughActiveParticipantsInChatError()

        super().__init__(name=chat.chat_leader.name, role=chat.chat_leader.role, **kwargs)

        self.inner_chat = chat
        self.spinner = spinner

    def on_new_chat_leader_elected(self, chat: 'ChatRoom', new_leader: Optional['ActiveChatParticipant']):
        if new_leader is None:
            return

        self.name = new_leader.name
        self.role = new_leader.role

    def respond_to_chat(self, chat: 'ChatRoom') -> str:
        if self.spinner is not None:
            self.spinner.stop_and_persist(symbol='👥', text=f'{self.name}\'s group started a discussion.')
            self.spinner.start(text=f'{self.name}\'s group is discussing...')

        messages = chat.chat_backing_store.get_messages()
        conversation_str = '\n'.join([f'- {message.sender_name}: {message.content}' for message in messages])
        response = self.inner_chat.initiate_chat_with_result(
            initial_message=f'''# ANOTHER EXTERNAL CONVERSATION\n{conversation_str}\n\n# TASK\nAs this group\'s leader, I need to respond in our group's name. What do you all think should I respond with? Let's collaborate on this.'''
        )

        if self.spinner is not None:
            self.spinner.succeed(text=f'{self.name}\'s group discussion was concluded.')

        return response


class JSONOutputParserChatParticipant(ActiveChatParticipant):
    output_schema: Type[TOutputSchema]
    output: Optional[TOutputSchema] = None

    def __init__(self,
                 output_schema: Type[TOutputSchema],
                 name: str = 'JSON Output Parser',
                 role: str = 'JSON Output Parser'
                 ):
        super().__init__(name=name, role=role)

        self.output_schema = output_schema

    def respond_to_chat(self, chat: 'ChatRoom') -> str:
        messages = chat.chat_backing_store.get_messages()
        if len(messages) == 0:
            raise NoMessagesInChatError()

        last_message = messages[-1]

        try:
            json_string = last_message.content[last_message.content.index('{'):last_message.content.rindex('}') + 1]
            self.output = model = json_string_to_pydantic(json_string, self.output_schema)

            return f'{model.model_dump_json()} TERMINATE'
        except Exception as e:
            return f'I could not parse the JSON. This was the error: {e}'


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
        mission='You will be provided some TEXT and a JSON SCHEMA. Your only mission is to convert the TEXT '
                'to a JSON that follows the JSON SCHEMA provided. Your message should include only correct JSON.',
        spinner=spinner
    )
    json_parser = JSONOutputParserChatParticipant(output_schema=output_schema)

    parser_chat = ChatRoom(
        initial_participants=[json_parser, text_to_json_ai],
        hide_messages=hide_message,
        max_total_messages=n_retries * 2
    )

    _ = parser_chat.initiate_chat_with_result(
        initial_message=f'# TEXT\n{output}\n\n# JSON SCHEMA\n{pydantic_to_json_schema(output_schema)}'
    )

    if json_parser.output is None:
        raise MessageCouldNotBeParsedError(output)

    return json_parser.output


if __name__ == '__main__':
    from langchain.chat_models import ChatOpenAI
    from dotenv import load_dotenv

    load_dotenv()
    chat_model = ChatOpenAI(temperature=0.0, model='gpt-4-0613')

    spinner = Halo(spinner='dots')

    # ai = AIChatParticipant(name='Assistant',
    #                        role='Boring Serious AI Assistant',
    #                        chat_model=chat_model,
    #                        spinner=spinner)
    # rob = AIChatParticipant(name='Rob', role='Funny Prankster',
    #                         mission='Collaborate with the user to prank the boring AI. Yawn.',
    #                         chat_model=chat_model,
    #                         spinner=spinner)
    # user = UserChatParticipant(name='User')
    # participants = [user, ai, rob]
    #
    # main_chat = ChatRoom(
    #     initial_participants=participants,
    #     chat_conductor=AIChatConductor(
    #         chat_model=chat_model,
    #         speaker_interaction_schema=f'Rob should take the lead and go back and forth with the assistant trying to prank him big time. Rob can and should talk to the user to get them in on the prank, however the majority of the prank should be done by Rob. By prank, I mean the AI should be confused and not know what to do, or laughs at the prank (funny).',
    #         termination_condition=f'Terminate the chat when the is successfuly pranked, or is unable to be pranked or does not go along with the pranks within a 2 tries, OR if the user asks you to terminate the chat.',
    #         spinner=spinner
    #     ),
    # )
    # main_chat.initiate_chat_with_result()

    # ai = AIChatParticipant(name='AI', role='Math Expert',
    #                        mission='Solve the user\'s math problem (only one). Respond with the correct answer and end with the word "TERMINATE"',
    #                        chat_model=chat_model, spinner=spinner)
    # user = UserChatParticipant(name='User')
    # participants = [user, ai]
    #
    #
    # class MathResult(BaseModel):
    #     result: float = Field(description='The result of the math problem.')
    #
    #
    # main_chat = ChatRoom(initial_participants=participants)
    # parsed_output = string_output_to_pydantic(
    #     output=main_chat.initiate_chat_with_result(),
    #     chat_model=chat_model,
    #     output_schema=MathResult,
    #     spinner=spinner
    # )
    #
    # print(f'Result: {dict(parsed_output)}')

    #     criteria_generation_team = GroupBasedChatParticipant(
    #         chat=ChatRoom(
    #             initial_participants=[
    #                 LangChainBasedAIChatParticipant(
    #                     name='Tom',
    #                     role='Criteria Generation Team Leader',
    #                     mission=f'Delegate to your team and respond back with comprehensive, orthogonal, well-researched criteria for a decision-making problem.',
    #                     other_prompt_sections={
    #                         'Last Message': '''- Once the criteria set is finalized you will send the last message.
    # - This last message will be sent to the external conversation verbatim. Act as if you are responding directly to the other chat yourself.
    # - Ignore the group and their efforts in the last message as this isn't relevant for the other chat.
    # '''
    #                     },
    #                     chat_model=chat_model,
    #                     spinner=spinner),
    #                 LangChainBasedAIChatParticipant(
    #                     name='Rob',
    #                     role='Criteria Generator',
    #                     mission='Think from first principles about the decision-making problem, and come up with orthogonal, compresive list of criteria. Iterate on it, as needed.',
    #                     other_prompt_sections={
    #                         'Receiving Feedback': 'John might criticize your criteria and provide counterfactual evidence to support his criticism. You should respond to his criticism and provide counter-counterfactual evidence to support your response, if applicable.'
    #                     },
    #                     chat_model=chat_model,
    #                     spinner=spinner),
    #                 LangChainBasedAIChatParticipant(
    #                     name='John',
    #                     role='Criteria Generation Critic',
    #                     mission='Think from frist principles and collaborate with Rob to come up with a comprehensive, orthogonal list of criteria. Criticize Rob\'s criteria and provide counterfactual evidence to support your criticism. Are some criteria overlapping and need to be merged? Is some criterion too general and need to be broken down? Are there criteria missing? Is the naming of each criteria suitable and reflects that a higher value is better? Iterate on it, as needed.',
    #                     other_prompt_sections={
    #                         'Receiving Feedback': 'Rob might criticize your criticism and provide counter-counterfactual evidence to support his response, if applicable.'
    #                     },
    #                     chat_model=chat_model,
    #                     spinner=spinner),
    #             ],
    #             chat_conductor=LangChainBasedAIChatConductor(
    #                 chat_model=chat_model,
    #                 speaker_interaction_schema='The team leader initiates the conversation about the criteria. Rob and John will go back and forth, refining and improving the criteria set until they both think the set cannot be improved anymore. Then, finally, once they both agree the set is good enough, the team leader responds with a message to the external conversation with the final criteria set.',
    #                 termination_condition='Terminate the chat when the team leader thinks the criteria set is good enough, or if the team leader asks you to terminate the chat.',
    #                 spinner=spinner
    #             ),
    #             description='This chat room is a group chat for the criteria generation team. Everybody can talk to everybody else. The goal is to generate a list of criteria for a decision-making problem.',
    #         ),
    #         spinner=spinner
    #     )
    #     user = UserChatParticipant(name='User')
    #     participants = [user, criteria_generation_team]
    #
    #     main_chat = ChatRoom(initial_participants=participants)
    #     result = main_chat.initiate_chat_with_result(
    #         initial_message="Please generate a list of criteria for choosing the breed of my next puppy.",
    #     ),
    #
    #     print(f'Result: {result}')

    from datetime import datetime
    from pydantic import Field


    def get_time():
        return datetime.utcnow().isoformat()


    class GetTime(BaseModel):
        """Returns the current time in UTC."""

        result: str = Field(description='The current time in UTC.')


    now = get_time()
    ai = LangChainBasedAIChatParticipant(name='Assistant',
                                         mission=f'Be a helpful, time-showing, AI assistant. Always tell them the UTC time (in a user friendly format) when you respond and how much time you have left to work. You stop working after 1 minute has passed from when you started working at {now}. When the time comes, let the user know you will stop working and say goodbye.',
                                         chat_model=chat_model,
                                         functions={
                                             GetTime.__name__: lambda _: get_time()
                                         },
                                         chat_model_args={
                                             'functions': [pydantic_to_openai_function(GetTime)]
                                         },
                                         spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai, logger]

    main_chat = ChatRoom(
        initial_participants=participants,
        chat_conductor=LangChainBasedAIChatConductor(
            chat_model=chat_model,
            termination_condition=f'Chat started at {now}. Terminate the chat when the assistant stops working (He will let it be known in the chat), OR if the user decides to terminate the chat by ending their sentence with "TERMINATE".',
            spinner=spinner
        ),
    )
    main_chat.initiate_chat_with_result()
