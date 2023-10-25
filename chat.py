import abc
import json
import re
import uuid
from collections import deque
from json import JSONDecodeError
from typing import List, Optional, Callable, Type, TypeVar, Dict, Tuple, Union

from halo import Halo
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, FunctionMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel, Field
from pydantic.v1 import ValidationError

from utils import fix_invalid_json

TOutputSchema = TypeVar("TOutputSchema", bound=Union[str, BaseModel])


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

    def send_message(self, chat: 'ChatRoom', content: str, recipient_name: str):
        if chat.id not in self._chats_joined:
            raise ChatParticipantNotJoinedToChat(self.name)

        chat.receive_message(sender_name=self.name, content=content, recipient_name=recipient_name)

    @staticmethod
    def get_recipient_and_message(message: str) -> Tuple[Optional[str], str]:
        pattern = r'\(?((.+?) to )?(.+?)\)?[#:](.+)'
        match = re.search(pattern, message, re.DOTALL)

        if match:
            return match.group(3).strip(), match.group(4).strip()

        return None, message.strip()

    def on_new_chat_messages(self, chat: 'ChatRoom', messages: List['ChatMessage'], termination_received: bool = False):
        raise NotImplementedError()

    def on_participant_joined_chat(self, chat: 'ChatRoom', participant: 'ChatParticipant'):
        if participant.name == self.name:
            self._chats_joined[chat.id] = chat

            return

    def on_participant_left_chat(self, chat: 'ChatRoom', participant: 'ChatParticipant'):
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
        super().__init__(f'Participant "{participant_name}" is not joined to this chat.')


class ChatParticipantAlreadyJoinedToChat(Exception):
    def __init__(self, participant_name: str):
        super().__init__(f'Participant "{participant_name}" is already joined to this chat.')


class MessageCouldNotBeParsed(Exception):
    def __init__(self, message: str):
        super().__init__(f'Message "{message}" could not be parsed.')


class ChatRoom:
    id: str
    messages: List[ChatMessage]
    participants: Dict[str, ChatParticipant]
    is_termination_message: Callable[[str], bool]
    description: str
    max_messages: Optional[int] = None
    last_message_id: Optional[int] = None
    hide_messages: bool = False
    _unprocessed_messages: deque

    def __init__(
            self,
            id: Optional[str] = None,
            initial_participants: Optional[List[ChatParticipant]] = None,
            initial_messages: Optional[List[ChatMessage]] = None,
            max_total_messages: Optional[int] = None,
            is_termination_message: Optional[Callable[[str], bool]] = None,
            hide_messages: bool = False,
            description: str = 'This chat room is a regular group chat. Everybody can talk to everybody else.'
    ):
        assert max_total_messages is None or max_total_messages > 0, 'Max total messages must be None or greater than 0.'

        self.id = str(uuid.uuid4()) if id is None else id
        self.participants = {}
        self.messages = sorted(initial_messages or [], key=lambda m: m.id)
        self.last_message_id = None if len(self.messages) == 0 else self.messages[-1].id
        self.is_termination_message = is_termination_message or (
            lambda message: message.content.strip().endswith('TERMINATE'))
        self._unprocessed_messages = deque()
        self.hide_messages = hide_messages
        self.max_messages = max_total_messages
        self.description = description

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

    def reset(self, remove_participants: bool = False):
        self.messages = []
        self.last_message_id = None
        self._unprocessed_messages = deque()

        if remove_participants:
            for participant in list(self.participants.values()):
                self.remove_participant(participant)

    def initiate_chat_with_result(
            self,
            first_message: str,
            from_participant: Union[str, ChatParticipant],
            to_participant: Union[str, ChatParticipant]
    ) -> str:
        if isinstance(to_participant, ChatParticipant):
            to_participant = to_participant.name

        if to_participant not in self.participants:
            raise ChatParticipantNotJoinedToChat(to_participant)

        if isinstance(from_participant, ChatParticipant):
            from_participant = from_participant.name

        if from_participant not in self.participants:
            raise ChatParticipantNotJoinedToChat(from_participant)

        sender = self.participants[from_participant]
        sender.send_message(chat=self, content=first_message, recipient_name=to_participant)

        while self._process_new_messages():
            pass

        result = self.messages[-1].content
        result = result.replace('TERMINATE', '').strip()

        return result

    def _process_new_messages(self) -> bool:
        termination_received = False

        new_messages = []
        while len(self._unprocessed_messages) > 0:
            message = self._unprocessed_messages.popleft()
            self.messages.append(message)

            sender = self.participants[message.sender_name]
            if not self.hide_messages and not sender.messages_hidden:
                self._display_new_message(message=message)

            new_messages.append(message)

            # Terminate if the max was reached, or if a termination message was received.
            # All messages after the termination message are dropped.
            if (self.max_messages is not None and len(self.messages) >= self.max_messages) or \
                    self.is_termination_message(message):
                termination_received = True

        for participant in self.participants.values():
            participant.on_new_chat_messages(chat=self, messages=new_messages,
                                             termination_received=termination_received)

        return len(new_messages) > 0 and not termination_received

    def _display_new_message(self, message: ChatMessage):
        if message.sender_name not in self.participants:
            symbol = '❓'
        else:
            symbol = self.participants[message.sender_name].symbol

        print(
            f'{symbol} ({message.sender_name} to {message.recipient_name}): {message.content}')


class UserChatParticipant(ChatParticipant):
    def __init__(self, name: str = 'User', **kwargs):
        super().__init__(name, role='User', messages_hidden=True, **kwargs)

    def on_new_chat_messages(self, chat: 'ChatRoom', messages: List['ChatMessage'], termination_received: bool = False):
        if termination_received:
            return

        relevant_messages = [message for message in messages if self.name == message.recipient_name]
        if len(relevant_messages) == 0:
            return

        new_message_contents = input(f'👤 ({self.name}): ')

        recipient_name, actual_message_contents = ChatParticipant.get_recipient_and_message(new_message_contents)
        recipient_name = recipient_name or relevant_messages[-1].sender_name

        self.send_message(chat=chat, content=actual_message_contents, recipient_name=recipient_name)


class AIChatParticipant(ChatParticipant):
    system_message_template: str = '''
# MISSION
- {mission}

# NAME
- {name}

# ROLE
- {role}

# CONVERSATION
- The conversation is a group chat you are a part of.
- Every participant can see all messages sent by all other participants, which means you cannot have private conversations with other participants.

# CHAT ROOM

## Description
{chat_room_description}

## Participants
{participants}

# INPUT
- Messages from the group chat, including your own messages.
  - They are prefixed by the sender's name and who the message is directed at (could also be everyone). For context only; it's not actually part of the message they sent.
  - Some messages could have been sent by participants who are no longer a part of this conversation. Use their contents for context only; do not respond to them.

# OUTPUT
- Your response to the user or other participants in the group chat.
- Always direct your message at one and only one (other than yourself) in the group chat.
- Every response you send should start with a recipient name followed by a hash (#) and then your message.
- The message after the # should not contain recipient name, as they are already specified before the #.
- A recipient MUST be one of the CURRENT participants in the group chat. However you CANNOT send a message to yourself.
- Do not prefix your message with (your name) to (recipient name) as that is already done for you.

# EXAMPLE OUTPUT
- When you want to respond prefix the recipient name with a # symbol and then the message like: RECIPIENT_NAME#YOUR_MESSAGE
- When you want to stay silent or ignore the message just respond with # like: #'''
    mission: str
    chat_model: ChatOpenAI
    spinner: Optional[Halo] = None
    can_terminate_conversation: bool = False
    termination_section: str = '''# TERMINATION
- You can terminate the conversation by sending a message ending with the word "TERMINATE".
- You should terminate the conversation when you have achieved your mission or goal OR when you are explicitly asked to terminate by another chat participant.'''

    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 name: str,
                 chat_model: ChatOpenAI,
                 symbol: str = '🤖',
                 role: str = 'AI Assistant',
                 mission: str = 'Be a helpful AI assistant.',
                 can_terminate_conversation: bool = True,
                 spinner: Optional[Halo] = None,
                 **kwargs
                 ):
        super().__init__(name=name, symbol=symbol, role=role, **kwargs)

        self.chat_model = chat_model
        self.spinner = spinner
        self.mission = mission
        self.can_terminate_conversation = can_terminate_conversation

    def _create_system_message(self, chat: ChatRoom):
        system_message = self.system_message_template.format(
            mission=self.mission,
            name=self.name,
            role=self.role,
            chat_room_description=chat.description,
            participants='\n'.join(
                [f'- Name: "{p.name}", Role: "{p.role}"{" -> This is you." if p.name == self.name else ""}' \
                 for p in chat.participants.values()])
        )

        if self.can_terminate_conversation:
            system_message += '\n\n' + self.termination_section

        return system_message

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

    def on_new_chat_messages(self, chat: 'ChatRoom', messages: List['ChatMessage'], termination_received: bool = False):
        if termination_received:
            return

        relevant_messages = [message for message in messages if self.name == message.recipient_name]
        if len(relevant_messages) == 0:
            return

        if self.spinner is not None:
            self.spinner.start(text=f'{self.name} ({self.role}) is thinking...')

        system_message = self._create_system_message(chat=chat)

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

        self.send_message(chat=chat, content=content, recipient_name=recipient_name)


class TeamBasedChatParticipant(ChatParticipant):
    team_leader: ChatParticipant
    other_team_participants: List[ChatParticipant]
    team_interaction_schema: Optional[str] = None
    hide_inner_chat: bool = True
    max_sub_chat_messages: Optional[int] = None
    team_chat_history_access: bool = True
    spinner: Optional[Halo] = None

    def __init__(self,
                 team_leader: ChatParticipant,
                 other_team_participants: List[ChatParticipant],
                 team_interaction_schema: Optional[str] = None,
                 hide_inner_chat: bool = True,
                 max_total_sub_chat_messages: Optional[int] = None,
                 team_chat_history_access: bool = True,
                 spinner: Optional[Halo] = None,
                 **kwargs):
        super().__init__(name=team_leader.name, role=team_leader.role, **kwargs)

        assert len(other_team_participants) >= 1, 'At least one team leader and 1 other team participant are required.'

        self.other_team_participants = other_team_participants
        self.team_leader = team_leader
        self.hide_inner_chat = hide_inner_chat
        self.max_sub_chat_messages = max_total_sub_chat_messages
        self.team_chat_history_access = team_chat_history_access
        self.team_interaction_schema = team_interaction_schema
        self.spinner = spinner

    def on_new_chat_messages(self, chat: 'ChatRoom', messages: List['ChatMessage'], termination_received: bool = False):
        if termination_received:
            return

        relevant_messages = [message for message in messages if self.name == message.recipient_name]
        if len(relevant_messages) == 0:
            return

        last_message = relevant_messages[-1]
        sender = chat.participants[last_message.sender_name]

        previous_sender_role = sender.role
        sender.role = 'Client'

        if self.spinner is not None:
            self.spinner.stop_and_persist(symbol='👥',
                                          text=f'{self.team_leader.name}\'s team started discussing the request.')
            self.spinner.start(text=f'{self.team_leader.name}\'s team is actively discussing the request...')

        sub_chat = ChatRoom(
            initial_participants=[self.team_leader, *self.other_team_participants, sender],
            initial_messages=chat.messages if self.team_chat_history_access else None,
            hide_messages=self.hide_inner_chat,
            max_total_messages=self.max_sub_chat_messages,
            description=f'The team leader is a part of another chat room as well as this one. '
                        f'This is a private team chat between {self.team_leader.name} and '
                        f'{", ".join([p.name for p in self.other_team_participants if p.role != "Client"])} ONLY. ' +
                        (
                            f'\n\n## Previous Chat Messages\nThe previous messages in this chat are from the other chat the team leader is a part of. '
                            f'No other participants can see the messages sent in this chat room other than the team members. '
                            f'Team members should ONLY respond to the team leader and other team members. '
                            f'The previous messages are there only for context. ' if self.team_chat_history_access else '') +
                        f'\n\n## Response To Client\nOnly the team leader will respond to client with the result of this chat room. '
                        f'The team should collaborate to come up with detailed and accurate response in the team leader\'s name. ' +
                        (
                            f'\n\n## Team Interaction Schema\n{self.team_interaction_schema}' if self.team_interaction_schema is not None else ''),
            is_termination_message=lambda message: message.content.strip().endswith(
                'TERMINATE') or message.recipient_name == sender.name
        )

        _, last_message_actual_content = self.get_recipient_and_message(last_message.content)

        response_from_team = sub_chat.initiate_chat_with_result(
            first_message=f'{last_message.sender_name} has told me this: "{last_message_actual_content}". What should I respond with?',
            from_participant=self.team_leader,
            to_participant=self.other_team_participants[0],
        )

        sender.role = previous_sender_role

        _, actual_message_contents = ChatParticipant.get_recipient_and_message(response_from_team)
        recipient_name = last_message.sender_name

        self.send_message(chat=chat, content=actual_message_contents, recipient_name=recipient_name)

        if self.spinner is not None:
            self.spinner.succeed(text=f'{self.team_leader.name}\'s team discussion was concluded.')


class JSONOutputParserChatParticipant(ChatParticipant):
    output_schema: Type[TOutputSchema]
    output: Optional[TOutputSchema] = None

    def __init__(self,
                 output_schema: Type[TOutputSchema],
                 name: str = 'JSON Output Parser',
                 role: str = 'JSON Output Parser'
                 ):
        super().__init__(name=name, role=role)

        self.output_schema = output_schema

    def on_new_chat_messages(self, chat: 'ChatRoom', messages: List['ChatMessage'], termination_received: bool = False):
        if termination_received:
            return

        relevant_messages = [message for message in messages if self.name == message.recipient_name]
        if len(relevant_messages) == 0:
            return

        last_message = relevant_messages[-1]

        if self.output_schema is str:
            self.send_message(chat=chat, content=f'{last_message.content} TERMINATE',
                              recipient_name=last_message.sender_name)
            self.output = last_message.content
        else:
            try:
                json_string = last_message.content[last_message.content.index('{'):last_message.content.rindex('}') + 1]
                self.output = model = json_string_to_pydantic(json_string, self.output_schema)

                self.send_message(chat=chat, content=f'{model.model_dump_json()} TERMINATE',
                                  recipient_name=last_message.sender_name)
            except Exception as e:
                self.send_message(chat=chat, content=f'I could not parse the JSON. This was the error: {e}',
                                  recipient_name=last_message.sender_name)


def string_output_to_pydantic(output: str, chat_model: ChatOpenAI, output_schema: Type[TOutputSchema],
                              spinner: Optional[Halo] = None,
                              n_retries: int = 3,
                              hide_message: bool = True) -> TOutputSchema:
    text_to_json_ai = AIChatParticipant(
        chat_model=chat_model,
        name='Text to JSON Converter',
        role='Text to JSON Converter',
        mission='You will be provided some TEXT and a JSON SCHEMA. Your only mission is to convert the TEXT '
                'to a JSON that follows the JSON SCHEMA provided. Your message should include only correct JSON.',
        spinner=spinner
    )
    json_parser = JSONOutputParserChatParticipant(output_schema=output_schema)

    parser_chat = ChatRoom(
        initial_participants=[text_to_json_ai, json_parser],
        hide_messages=hide_message,
        max_total_messages=n_retries * 2
    )

    _ = parser_chat.initiate_chat_with_result(
        first_message=f'# TEXT\n{output}\n\n# JSON SCHEMA\n{pydantic_to_json_schema(output_schema)}',
        from_participant=json_parser,
        to_participant=text_to_json_ai
    )

    if json_parser.output is None:
        raise MessageCouldNotBeParsed(output)

    return json_parser.output


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    chat_model = ChatOpenAI(temperature=0.0, model='gpt-4-0613')

    spinner = Halo(spinner='dots')

    # ai = AIChatParticipant(name='Assistant', chat_model=chat_model, spinner=spinner)
    # rob = AIChatParticipant(name='Rob', role='Funny Prankster',
    #                         mission='Collaborate with the user to prank the boring AI. Yawn.',
    #                         chat_model=chat_model, spinner=spinner)
    # user = UserChatParticipant(name='User')
    # participants = [ai, rob, user]

    # ai = AIChatParticipant(name='AI', role='Math Expert',
    #                        mission='Solve the user\'s math problem and TERMINATE immediately (only if you have the solution). The user has only one problem.',
    #                        chat_model=chat_model, spinner=spinner)
    # user = UserChatParticipant(name='User')
    # participants = [ai, user]
    #
    #
    # class MathResult(BaseModel):
    #     result: float = Field(description='The result of the math problem.')
    #
    #
    # main_chat = ChatRoom(initial_participants=participants)
    # parsed_output = string_output_to_pydantic(
    #     output=main_chat.initiate_chat_with_result(
    #         first_message="Hey",
    #         from_participant=user,
    #         to_participant=ai
    #     ),
    #     chat_model=chat_model,
    #     output_schema=MathResult,
    #     spinner=spinner
    # )

    ai = AIChatParticipant(name='Assistant',
                           mission=f'Be a helpful AI assistant. However, do not answer any questions about geography yourself; instead, ask the user to ask the research team, and only then respond with the answer the research team came up with.',
                           chat_model=chat_model,
                           spinner=spinner)
    research_team = TeamBasedChatParticipant(
        team_leader=AIChatParticipant(name='Research Team Leader',
                                      role='Research Team Leader',
                                      mission=f'Respond back with a great well-researched answer to the question asked of you. Do this by collaborating with the internal research team to answer questions. Respond as if you came up with the answer yourself. TERMINATE immediately when the research team gives you the answer.',
                                      chat_model=chat_model,
                                      spinner=spinner),
        other_team_participants=[
            AIChatParticipant(name='Lead Researcher',
                              role='Researcher',
                              mission='Research whatever was asked of you by the research team leader. Come up with an initial hypothesis and validate it with the other researchers; iterate on it until you all agree on a good answer. Once you have a good answer, send it to the team leader.',
                              chat_model=chat_model,
                              can_terminate_conversation=False,
                              spinner=spinner),
            AIChatParticipant(name='Researcher 1',
                              role='Researcher',
                              mission='Collaborate with the researchers in the team to come up with the best answer. Come up with counterfactual evidence, try to disprove the hypothesis, and validate the hypothesis with the other researchers; iterate on it, until you all agree on a good answer.',
                              chat_model=chat_model,
                              can_terminate_conversation=False,
                              spinner=spinner),
        ],
        team_interaction_schema='The team leader will ask the lead researcher to come up with an answer. The lead researcher will draft an hypothesis and validate it with the other researcher. They will go back and forth, providing counterfactual evidence and validating it with each other. Once they have a good answer, the lead researcher will send it to the team leader, who will then send it back to the client.',
        hide_inner_chat=False,
        spinner=spinner
    )
    user = UserChatParticipant(name='User')
    participants = [ai, research_team, user]

    main_chat = ChatRoom(initial_participants=participants)
    result = main_chat.initiate_chat_with_result(
        first_message="Hey",
        from_participant=user,
        to_participant=ai
    ),

    print(f'Result: {result}')
