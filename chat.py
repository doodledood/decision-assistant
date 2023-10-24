import json
from json import JSONDecodeError
from typing import List, Optional, Callable, Type, TypeVar, Dict

from halo import Halo
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, FunctionMessage, HumanMessage
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


class ChatParticipant(BaseModel):
    name: str
    _chats_joined = Dict[int, 'Chat']

    def __init__(self, name: str):
        super().__init__(name=name)

        self._chats_joined = {}

    def send_message(self, chat: 'Chat', content: str):
        chat.receive_message(sender=self, content=content)

    def join_chat(self, chat: 'Chat'):
        chat.add_participant(participant=self)

        self._chats_joined[chat.get_id()] = chat

    def leave_chat(self, chat: 'Chat'):
        chat_id = chat.get_id()
        if chat_id not in self._chats_joined:
            raise ChatParticipantNotJoinedToChat(self)

        chat.remove_participant(participant=self)

        self._chats_joined.pop(chat_id)

    def on_new_chat_message(self, chat: 'Chat', message: 'ChatMessage'):
        pass

    def on_removed_from_chat(self, chat: 'Chat'):
        chat_id = chat.get_id()

        if chat_id in self._chats_joined:
            self._chats_joined.pop(chat_id)

    def __hash__(self):
        return hash(self.name)


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
    _id: int
    _messages: List[ChatMessage]
    _participants: Dict[str, ChatParticipant]
    _last_message_id: Optional[int]

    def __init__(
            self,
            id: int,
            initial_participants: Optional[List[ChatParticipant]] = None,
            initial_messages: Optional[List[ChatMessage]] = None
    ):
        self._id = id
        self._participants = {}
        self._messages = sorted(initial_messages or [], key=lambda m: m.id)
        self._last_message_id = None if len(self._messages) == 0 else self._messages[-1].id

        for participant in initial_participants or []:
            self.add_participant(participant)

    def get_id(self):
        return self._id

    def get_participants(self):
        return self._participants

    def add_participant(self, participant: ChatParticipant):
        if participant.name in self._participants:
            raise ChatParticipantAlreadyJoinedToChat(participant)

        self._participants[participant.name] = participant

    def remove_participant(self, participant: ChatParticipant):
        if participant.name not in self._participants:
            raise ChatParticipantNotJoinedToChat(participant)

        self._participants.pop(participant.name)

        participant.on_removed_from_chat(chat=self)

    def receive_message(self, sender: ChatParticipant, content: str):
        if sender not in self._participants:
            raise ChatParticipantNotJoinedToChat(sender)

        self._last_message_id = self._last_message_id + 1 if self._last_message_id is not None else 1

        self._messages.append(ChatMessage(
            id=self._last_message_id,
            sender=sender,
            content=content
        ))

        for participant in self._participants.values():
            if participant.name == sender.name:
                continue

            participant.on_new_chat_message(chat=self, message=self._messages[-1])

# participant1 = ChatParticipant(name='Participant 1')
# participant2 = ChatParticipant(name='Participant 2')
#
# chat = Chat(id=1, initial_participants=[participant1, participant2])
#
# participant1.send_message(chat, 'Hello!')
