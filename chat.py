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
    role: str
    messages_hidden: bool
    _chats_joined = Dict[int, 'Chat']

    def __init__(self, name: str, symbol: str = '👤', role: str = 'Chat Participant', messages_hidden: bool = False):

        self.name = name
        self.symbol = symbol
        self.role = role
        self.messages_hidden = messages_hidden

        self._chats_joined = {}

    def send_message(self, chat: 'Chat', content: str, recipient_names: Optional[List[str]] = None):
        if chat.id not in self._chats_joined:
            raise ChatParticipantNotJoinedToChat(self.name)

        chat.receive_message(sender_name=self.name, content=content, recipient_names=recipient_names)

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
    recipient_names: Optional[List[str]]
    content: str


class ChatParticipantNotJoinedToChat(Exception):
    def __init__(self, participant_name: str):
        super().__init__(f'Participant {participant_name} is not joined to this chat.')


class ChatParticipantAlreadyJoinedToChat(Exception):
    def __init__(self, participant_name: str):
        super().__init__(f'Participant {participant_name} is already joined to this chat.')


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
            raise ChatParticipantAlreadyJoinedToChat(participant.name)

        self.participants[participant.name] = participant

        for participant in self.participants.values():
            participant.on_participant_joined_chat(chat=self, participant=participant)

    def remove_participant(self, participant: ChatParticipant):
        if participant.name not in self.participants:
            raise ChatParticipantNotJoinedToChat(participant.name)

        self.participants.pop(participant.name)

        for participant in self.participants.values():
            participant.on_participant_left_chat(chat=self, participant=participant)

    def receive_message(self, sender_name: str, content: str, recipient_names: Optional[List[str]] = None):
        if sender_name not in self.participants:
            raise ChatParticipantNotJoinedToChat(sender_name)

        sender = self.participants[sender_name]

        self.last_message_id = self.last_message_id + 1 if self.last_message_id is not None else 1

        if recipient_names is None:
            recipient_names = [participant.name for participant in self.participants.values() if
                               participant.name != sender_name]

        message = ChatMessage(
            id=self.last_message_id,
            sender_name=sender.name,
            recipient_names=recipient_names,
            content=content
        )

        self.messages.append(message)

        if not sender.messages_hidden:
            self.display_new_message(message=message)

        for participant in self.participants.values():
            participant.on_new_chat_message(chat=self, message=message)

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
        if message.sender_name == self.name or (
                message.recipient_names is not None and self.name not in message.recipient_names):
            return

        new_message_contents = input(f'👤 ({self.name}): ')

        parts = new_message_contents.split('#', maxsplit=1)
        if len(parts) == 2:
            recipients, actual_message_contents = parts
            recipients = [recipient.strip() for recipient in recipients.split(',')]
            actual_message_contents = actual_message_contents.strip()
        else:
            recipients, actual_message_contents = None, new_message_contents

        self.send_message(chat=chat, content=actual_message_contents, recipient_names=recipients)


class AIChatParticipant(ChatParticipant):
    system_message: str = '''
# MISSION
- {mission}

# NAME
- {name}

# ROLE
- {role}

# CONVERSATION
- The conversation is a group chat you are a part of.
- Every participant can see all messages sent by all other participants, which means you cannot have private conversations with other participants.

# CHAT PARTICIPANTS
{participants}

# INPUT
- Messages from the user or other participants in the group chat.
- The messages represent previous messages from the group chat you are a part of.
- They are prefixed by the sender's name.

# STAYING SILENT
- Sometimes when conversing with other participants, you may want to stay silent or ignore a message. For example, when you are in a group chat with more than 1 participant, and 2 of you are talking with the other participant, you may want to stay silent when the other participant is talking to the other participant, unless you are explicitly mentioned.

# OUTPUT
- Your response to the user or other participants in the group chat.
- Do not prefix your messages with your name. Assume the participants know who you are.
- Always direct your message at one or more participants (other than yourself) in the group chat.
- Every response you send should start with a recipient name followed by a hash (#) and then your message.
- However, you do have the option to not respond to a message, in which case you should send an empty message (i.e. just a hash (#)).

# EXAMPLE OUTPUT
- When you want to respond: RECIPIENT1_NAME,...,RECIPIENT2_NAME#YOUR_MESSAGE
- When you want to stay silent or ignore the message: #
'''
    mission: str
    chat_model: ChatOpenAI
    spinner: Optional[Halo] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 name: str,
                 chat_model: ChatOpenAI,
                 symbol: str = '🤖',
                 role: str = 'AI Assistant',
                 mission: str = 'Be a helpful AI assistant.',
                 system_message: Optional[str] = None,
                 spinner: Optional[Halo] = None
                 ):
        super().__init__(name=name, symbol=symbol, role=role)

        self.chat_model = chat_model
        self.system_message = system_message or self.system_message
        self.spinner = spinner
        self.mission = mission

    def _chat_messages_to_chat_model_messages(self, chat_messages: List[ChatMessage]) -> List[BaseMessage]:
        messages = []
        for message in chat_messages:
            if message.sender_name == self.name:
                messages.append(AIMessage(content=message.content))
            else:
                messages.append(HumanMessage(content=f'({message.sender_name}): {message.content}'))

        return messages

    def on_new_chat_message(self, chat: 'Chat', message: 'ChatMessage'):
        if message.sender_name == self.name or (
                message.recipient_names is not None and self.name not in message.recipient_names):
            return

        if self.spinner is not None:
            self.spinner.start(text=f'{self.name} ({self.role}) is thinking...')

        system_message = self.system_message.format(
            mission=self.mission,
            name=self.name,
            role=self.role,
            participants='\n'.join([f'- {p.name} ({p.role}){" -> This is you." if p.name == self.name else ""}' \
                                    for p in chat.participants.values()])
        )

        all_messages = self._chat_messages_to_chat_model_messages(chat.messages)
        all_messages = [
            SystemMessage(content=system_message),
            *all_messages
        ]

        last_message = self.chat_model.predict_messages(all_messages)

        if self.spinner is not None:
            self.spinner.stop()

        parts = last_message.content.split('#', maxsplit=1)
        if len(parts) == 2:
            recipients, content = parts
            recipients = [recipient.strip() for recipient in recipients.split(',')]
            content = content.strip()
        else:
            recipients, content = None, last_message.content

        if content == '':
            return

        self.send_message(chat=chat, content=content, recipient_names=recipients)


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    chat_model = ChatOpenAI(temperature=0.0, model='gpt-4-0613')

    spinner = Halo(spinner='dots')
    ai = AIChatParticipant(name='Assistant', chat_model=chat_model, spinner=spinner)
    rob = AIChatParticipant(name='Rob', role='Funny Prankster',
                            mission='Collaborate with the user to prank the boring AI. Yawn.',
                            chat_model=chat_model, spinner=spinner)
    user = UserChatParticipant(name='User')

    main_chat = Chat(initial_participants=[ai, rob, user])
    user.send_message(chat=main_chat, content='Hello!')
