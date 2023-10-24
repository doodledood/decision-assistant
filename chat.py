import abc
import json
import re
import uuid
from collections import deque
from json import JSONDecodeError
from typing import List, Optional, Callable, Type, TypeVar, Dict, ClassVar, Tuple

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

    def send_message(self, chat: 'Chat', content: str, recipient_name: str):
        if chat.id not in self._chats_joined:
            raise ChatParticipantNotJoinedToChat(self.name)

        chat.receive_message(sender_name=self.name, content=content, recipient_name=recipient_name)

    @staticmethod
    def get_recipient_and_message(message: str) -> Tuple[Optional[str], str]:
        pattern1 = r'\((.+?) to (.+?)\):\s*(.+)'
        match1 = re.search(pattern1, message)

        if match1:
            return match1.group(2).strip(), match1.group(3).strip()

        pattern2 = r'(.+?)#(.+)'
        match2 = re.search(pattern2, message)
        if match2:
            return match2.group(1).strip(), match2.group(2).strip()

        pattern3 = r'\((.+?) to (.+?)\)#\s*(.+)'
        match3 = re.search(pattern3, message)

        if match3:
            return match1.group(2).strip(), match1.group(3).strip()

        return None, message.strip()

    def on_new_chat_messages(self, chat: 'Chat', messages: List['ChatMessage']):
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
    recipient_name: str
    content: str


class ChatParticipantNotJoinedToChat(Exception):
    def __init__(self, participant_name: str):
        super().__init__(f'Participant {participant_name} is not joined to this chat.')


class ChatParticipantAlreadyJoinedToChat(Exception):
    def __init__(self, participant_name: str):
        super().__init__(f'Participant {participant_name} is already joined to this chat.')


class MessageCouldNotBeParsed(Exception):
    def __init__(self, message: str):
        super().__init__(f'Message "{message}" could not be parsed.')


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
        self._unprocessed_messages = deque()

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

    def receive_message(self, sender_name: str, content: str, recipient_name: str):
        if sender_name not in self.participants:
            raise ChatParticipantNotJoinedToChat(sender_name)

        if recipient_name not in self.participants:
            raise ChatParticipantNotJoinedToChat(recipient_name)

        sender = self.participants[sender_name]

        self.last_message_id = self.last_message_id + 1 if self.last_message_id is not None else 1

        message = ChatMessage(
            id=self.last_message_id,
            sender_name=sender.name,
            recipient_name=recipient_name,
            content=content
        )

        self._unprocessed_messages.append(message)
       
    def run(self):
        while self._process_new_messages():
            pass

    def _process_new_messages(self) -> bool:
        new_messages = []
        while len(self._unprocessed_messages) > 0:
            message = self._unprocessed_messages.popleft()
            self.messages.append(message)

            sender = self.participants[message.sender_name]
            if not sender.messages_hidden:
                self._display_new_message(message=message)

            new_messages.append(message)

        for participant in self.participants.values():
            participant.on_new_chat_messages(chat=self, messages=new_messages)

        return len(new_messages) > 0

    def _display_new_message(self, message: ChatMessage):
        if message.sender_name not in self.participants:
            symbol = '❓'
        else:
            symbol = self.participants[message.sender_name].symbol

        print(
            f'{symbol} ({message.sender_name} to {message.recipient_name}): {message.content}')


class UserChatParticipant(ChatParticipant):
    def __init__(self, name: str = 'User'):
        super().__init__(name, role='User', messages_hidden=True)

    def on_new_chat_messages(self, chat: 'Chat', messages: List['ChatMessage']):
        relevant_messages = [message for message in messages if self.name == message.recipient_name]
        if len(relevant_messages) == 0:
            return

        new_message_contents = input(f'👤 ({self.name}): ')

        recipient_name, actual_message_contents = ChatParticipant.get_recipient_and_message(new_message_contents)
        recipient_name = recipient_name or relevant_messages[-1].sender_name

        self.send_message(chat=chat, content=actual_message_contents, recipient_name=recipient_name)


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
- Messages from the group chat, including your own messages.
  - They are prefixed by the sender's name and who the message is directed at (could also be everyone). For context only; it's not actually part of the message they sent.
- They are prefixed by the sender's name and who the message is directed at (could also be everyone).

# STAYING SILENT
- Sometimes when conversing with other participants, you may want to stay silent or ignore a message. For example, when you are in a group chat with more than 1 participant, and 2 of you are talking with the other participant, you may want to stay silent when the other participant is talking to the other participant, unless you are explicitly mentioned.

# OUTPUT
- Your response to the user or other participants in the group chat.
- Always direct your message at one and only one (other than yourself) in the group chat.
- Every response you send should start with a recipient name followed by a hash (#) and then your message.
- The message after the # should not contain recipient name, as they are already specified before the #.
- However, you do have the option to not respond to a message, in which case you should send an empty message (i.e. just a hash (#)).
- Do not prefix your message with (your name) to (recipient name) as that is already done for you.

# EXAMPLE OUTPUT
- When you want to respond prefix the recipient name with a # symbol and then the message like: RECIPIENT_NAME#YOUR_MESSAGE
- When you want to stay silent or ignore the message just respond with # like: #
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
            content = \
                f'({message.sender_name} to {message.recipient_name}): {message.content}'
            if message.sender_name == self.name:
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        return messages

    def on_new_chat_messages(self, chat: 'Chat', messages: List['ChatMessage']):
        relevant_messages = [message for message in messages if self.name == message.recipient_name]
        if len(relevant_messages) == 0:
            return

        if self.spinner is not None:
            self.spinner.start(text=f'{self.name} ({self.role}) is thinking...')

        system_message = self.system_message.format(
            mission=self.mission,
            name=self.name,
            role=self.role,
            participants='\n'.join(
                [f'- Name: "{p.name}", Role: "{p.role}"{" -> This is you." if p.name == self.name else ""}' \
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

        recipient_name, content = ChatParticipant.get_recipient_and_message(last_message.content)
        recipient_name = recipient_name or relevant_messages[-1].sender_name

        if content == '':
            return

        self.send_message(chat=chat, content=content, recipient_name=recipient_name)


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
    user.send_message(chat=main_chat, content='Hello!', recipient_name='Assistant')

    main_chat.run()
