import abc
import json
import uuid
from json import JSONDecodeError
from typing import List, Optional, Callable, Type, TypeVar, Dict, ClassVar

from halo import Halo
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, FunctionMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel
from pydantic.v1 import ValidationError

from utils import fix_invalid_json

ResultSchema = TypeVar("T", bound=BaseModel)


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


def chat(chat_model: ChatOpenAI,
         messages: List[BaseMessage],
         tools: Optional[List[Tool]] = None,
         get_user_input: Optional[Callable[[List[BaseMessage]], str]] = None,
         on_reply: Optional[Callable[[str], None]] = None,
         result_schema: Optional[Type[BaseModel]] = None,
         spinner: Optional[Halo] = None,
         max_ai_messages: Optional[int] = None,
         max_consecutive_arg_error_count: int = 3,
         get_immediate_answer: bool = False,
         ) -> ResultSchema:
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
    symbol: str
    messages_hidden: bool
    _chats_joined = Dict[int, 'Chat']

    def __init__(self, name: str, symbol: str = '👤', messages_hidden: bool = False):

        self.name = name
        self.symbol = symbol
        self.messages_hidden = messages_hidden

        self._chats_joined = {}

    def send_message(self, chat: 'Chat', content: str):
        if chat.id not in self._chats_joined:
            raise ChatParticipantNotJoinedToChat(self)

        chat.receive_message(sender=self, content=content)

    def on_new_chat_message(self, chat: 'Chat', message: 'ChatMessage'):
        raise NotImplementedError()

    def on_participant_joined_chat(self, chat: 'Chat', participant: 'ChatParticipant'):
        if participant.name == self.name:
            self._chats_joined[chat.id] = chat

            return

    def on_participant_left_chat(self, chat: 'Chat', participant: 'ChatParticipant'):
        if participant.name == self.name:
            self._chats_joined.pop(chat.id)

            return


class ChatMessage(BaseModel):
    id: int
    sender_name: str
    content: str


class ChatParticipantNotJoinedToChat(Exception):
    def __init__(self, participant: ChatParticipant):
        super().__init__(f'Participant {participant} is not joined to this chat.')


class ChatParticipantAlreadyJoinedToChat(Exception):
    def __init__(self, participant: ChatParticipant):
        super().__init__(f'Participant {participant} is already joined to this chat.')


class Chat:
    id: str
    messages: List[ChatMessage]
    participants: Dict[str, ChatParticipant]
    last_message_id: Optional[int]

    def __init__(
            self,
            id: Optional[str] = None,
            initial_participants: Optional[List[ChatParticipant]] = None,
            initial_messages: Optional[List[ChatMessage]] = None
    ):
        self.id = str(uuid.uuid4()) if id is None else id
        self.participants = {}
        self.messages = sorted(initial_messages or [], key=lambda m: m.id)
        self.last_message_id = None if len(self.messages) == 0 else self.messages[-1].id

        for participant in initial_participants or []:
            self.add_participant(participant)

    def add_participant(self, participant: ChatParticipant):
        if participant.name in self.participants:
            raise ChatParticipantAlreadyJoinedToChat(participant)

        self.participants[participant.name] = participant

        for participant in self.participants.values():
            participant.on_participant_joined_chat(chat=self, participant=participant)

    def remove_participant(self, participant: ChatParticipant):
        if participant.name not in self.participants:
            raise ChatParticipantNotJoinedToChat(participant)

        self.participants.pop(participant.name)

        for participant in self.participants.values():
            participant.on_participant_left_chat(chat=self, participant=participant)

    def receive_message(self, sender: ChatParticipant, content: str):
        if sender.name not in self.participants:
            raise ChatParticipantNotJoinedToChat(sender)

        self.last_message_id = self.last_message_id + 1 if self.last_message_id is not None else 1

        message = ChatMessage(
            id=self.last_message_id,
            sender_name=sender.name,
            content=content
        )

        self.messages.append(message)

        if not sender.messages_hidden:
            self.display_new_message(message=message)

        for participant in self.participants.values():
            participant.on_new_chat_message(chat=self, message=self.messages[-1])

    def display_new_message(self, message: ChatMessage):
        if message.sender_name not in self.participants:
            symbol = '❓'
        else:
            symbol = self.participants[message.sender_name].symbol

        print(f'{symbol} ({message.sender_name}): {message.content}')


class UserChatParticipant(ChatParticipant):
    def __init__(self, name: str = 'User'):
        super().__init__(name, messages_hidden=True)

    def on_new_chat_message(self, chat: 'Chat', message: 'ChatMessage'):
        if message.sender_name == self.name:
            return

        new_message_contents = input(f'👤 ({self.name}): ')

        self.send_message(chat=chat, content=new_message_contents)


class AIChatParticipant(ChatParticipant):
    system_message: str = '''
    # MISSION
    - {mission}
    
    # NAME
    - {name}
    
    # ROLE
    - {role}
    
    # INPUT
    - Messages from the user.
    - The messages represent previous messages from the group chat you are a part of.
    - They are prefixed by the sender's name.
    
    # OUTPUT
    - Your response to the user or other participants in the group chat.
    - Do not prefix your messages with your name. Assume the participants know who you are.
    
    # TERMINATION
    - When you think you have achieved your mission or goal, please respond with final result of your mission or goal.
    - End the message with TERMINATE.
    
    # TERMINATION EXAMPLE
    - The result of my mission or goal. TERMINATE
    '''
    mission: str
    role: str
    chat_model: ChatOpenAI
    spinner: Optional[Halo] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 name: str,
                 chat_model: ChatOpenAI,
                 symbol: str = '🤖',
                 mission: str = 'Be a helpful AI assistant.',
                 role='AI Assistant',
                 system_message: Optional[str] = None,
                 spinner: Optional[Halo] = None
                 ):
        super().__init__(name=name, symbol=symbol)

        self.chat_model = chat_model
        self.system_message = system_message or self.system_message
        self.spinner = spinner
        self.mission = mission
        self.role = role

        self.system_message = self.system_message.format(name=self.name, mission=self.mission, role=self.role)

    def _chat_messages_to_chat_model_messages(self, chat_messages: List[ChatMessage]) -> List[BaseMessage]:
        messages = []
        for message in chat_messages:
            if message.sender_name == self.name:
                messages.append(AIMessage(content=message.content))
            else:
                messages.append(HumanMessage(content=f'({message.sender_name}): {message.content}'))

        return messages

    def on_new_chat_message(self, chat: 'Chat', message: 'ChatMessage'):
        if message.sender_name == self.name:
            return

        if self.spinner is not None:
            self.spinner.start(text='Thinking...')

        all_messages = self._chat_messages_to_chat_model_messages(chat.messages)
        all_messages = [
            SystemMessage(content=self.system_message),
            *all_messages
        ]

        last_message = self.chat_model.predict_messages(all_messages)

        if self.spinner is not None:
            self.spinner.stop()

        self.send_message(chat=chat, content=last_message.content)


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    chat_model = ChatOpenAI(temperature=0.0, model='gpt-4-0613')

    spinner = Halo(spinner='dots')
    ai = AIChatParticipant(name='AI Assistant', chat_model=chat_model, spinner=spinner)
    user = UserChatParticipant(name='User')

    main_chat = Chat(initial_participants=[ai, user])
    user.send_message(chat=main_chat, content='Hello!')
