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

    def __init__(self, name: str, symbol: str = '👤', role: str = 'Chat Participant', messages_hidden: bool = False):
        self.name = name
        self.symbol = symbol
        self.role = role
        self.messages_hidden = messages_hidden

    @abc.abstractmethod
    def respond_to_chat(self, chat: 'ChatRoom') -> str:
        raise NotImplementedError()

    def on_new_chat_message(self, chat: 'ChatRoom', message: 'ChatMessage'):
        pass

    def on_chat_ended(self, chat: 'ChatRoom'):
        pass

    def on_participant_joined_chat(self, chat: 'ChatRoom', participant: 'ChatParticipant'):
        pass

    def on_participant_left_chat(self, chat: 'ChatRoom', participant: 'ChatParticipant'):
        pass

    def on_new_chat_leader_elected(self, chat: 'ChatRoom', new_leader: Optional['ChatParticipant']):
        pass


class ChatMessage(BaseModel):
    id: int
    sender_name: str
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


class NotEnoughParticipantsInChat(Exception):
    def __init__(self, n_participants: int = 0):
        super().__init__(f'There are not enough participants in this chat ({n_participants})')


class NoMessagesInChat(Exception):
    def __init__(self):
        super().__init__('There are no messages in this chat.')


class ChatConductor(abc.ABC):
    @abc.abstractmethod
    def select_next_speaker(self, chat: 'ChatRoom') -> Optional[ChatParticipant]:
        raise NotImplementedError()

    def get_chat_result(self, chat: 'ChatRoom') -> str:
        if len(chat.messages) == 0:
            return ''

        last_message = chat.messages[-1]

        return last_message.content

    def elect_new_chat_leader(self, chat: 'ChatRoom') -> ChatParticipant:
        if len(chat.participants) == 0:
            raise NotEnoughParticipantsInChat()

        return next(iter(chat.participants.values()))


class BasicChatConductor(ChatConductor):
    def select_next_speaker(self, chat: 'ChatRoom') -> Optional[ChatParticipant]:
        if len(chat.participants) <= 1:
            return None

        last_message = chat.messages[-1] if len(chat.messages) > 0 else None

        if last_message is not None and self.is_termination_message(last_message):
            return None

        last_speaker = last_message.sender_name if last_message is not None else None
        if last_speaker is None:
            return next(iter(chat.participants.values()))

        # Rotate to the next participant in the list.
        participant_names = list(chat.participants.keys())
        last_speaker_index = participant_names.index(last_speaker)
        next_speaker_index = (last_speaker_index + 1) % len(participant_names)
        next_speaker_name = participant_names[next_speaker_index]
        next_speaker = chat.participants[next_speaker_name]

        return next_speaker

    def get_chat_result(self, chat: 'ChatRoom') -> str:
        result = super().get_chat_result(chat=chat)
        result = result.replace('TERMINATE', '').strip()

        return result

    def is_termination_message(self, message: ChatMessage):
        return message.content.strip().endswith('TERMINATE')


class AIChatConductor(ChatConductor):
    chat_model: ChatOpenAI
    speaker_interaction_schema: Optional[str]
    spinner: Optional[Halo] = None

    def __init__(self, chat_model: ChatOpenAI, speaker_interaction_schema: Optional[str] = None, spinner: Optional[Halo] = None):
        self.chat_model = chat_model
        self.speaker_interaction_schema = speaker_interaction_schema
        self.spinner = spinner

    def select_next_speaker(self, chat: 'ChatRoom') -> Optional[ChatParticipant]:
        if len(chat.participants) <= 1:
            return None

        if self.spinner is not None:
            self.spinner.start(text='AI Conductor is selecting the next speaker...')

        # Ask the AI to select the next speaker.
        messages_str = '\n'.join(
            [f'- {message.sender_name}: {message.content}' for message in chat.messages])
        messages = [
            SystemMessage(content=f'''# MISSION
- Select the next speaker in the conversation based on the previous messages in the conversation and an optional interaction schema.
- If it seems to you that the chat should end instead of selecting a next speaker, terminate it.

# PARTICIPANTS
{', '.join([participant.name for participant in chat.participants.values()])}

# RULES
- You can only select one of the participants in the group chat.

# SPEAKER INTERACTION SCHEMA
{self.speaker_interaction_schema or 'Not provided.'}

# PROCESS
- Look at the last message in the conversation and determine who should speak next based on the speaker interaction schema, if provided.
- If, based on the interaction schema, it seems to you that the chat should end instead, you should return the string TERMINATE instead of a participant name.

# INPUT
- The previous messages in the conversation.

# OUTPUT
- The name of the next speaker in the conversation. Or, TERMINATE if the chat should end, instead.

# EXAMPLE OUTPUTS
- "John"
OR
- "TERMINATE"'''),
            HumanMessage(content=f'# PREVIOUS MESSAGES\n\n{messages_str if len(messages_str) > 0 else "No messages yet."}')
        ]

        result = self.chat_model.predict_messages(messages)
        next_speaker_name = result.content.strip()

        while next_speaker_name not in chat.participants and next_speaker_name != 'TERMINATE':
            messages.append(result)
            messages.append(HumanMessage(
                content=f'Speaker "{next_speaker_name}" is not a participant in the chat. Choose another one.'))

            result = self.chat_model.predict_messages(messages)
            next_speaker_name = result.content.strip()

        if next_speaker_name == 'TERMINATE':
            if self.spinner is not None:
                self.spinner.stop_and_persist(symbol='👥', text='AI Conductor has decided to terminate the chat.')

            return None

        next_speaker = chat.participants[next_speaker_name]

        if self.spinner is not None:
            self.spinner.succeed(text=f'AI Conductor has selected "{next_speaker_name}" as the next speaker.')

        return next_speaker


class ChatRenderer(abc.ABC):
    def render_new_chat_message(self, chat: 'ChatRoom', message: ChatMessage):
        raise NotImplementedError()


class TerminalChatRenderer(ChatRenderer):
    def render_new_chat_message(self, chat: 'ChatRoom', message: ChatMessage):
        if chat.hide_messages:
            return

        if message.sender_name not in chat.participants:
            symbol = '❓'
        else:
            participant = chat.participants[message.sender_name]
            if participant.messages_hidden:
                return

            symbol = participant.symbol

        print(f'{symbol} {message.sender_name}: {message.content}')


class ChatRoom:
    messages: List[ChatMessage]
    participants: Dict[str, ChatParticipant]
    chat_conductor: ChatConductor
    chat_renderer: ChatRenderer
    description: str
    chat_leader: Optional[ChatParticipant] = None
    max_messages: Optional[int] = None
    last_message_id: Optional[int] = None
    hide_messages: bool = False

    def __init__(
            self,
            initial_participants: Optional[List[ChatParticipant]] = None,
            initial_messages: Optional[List[ChatMessage]] = None,
            chat_conductor: Optional[ChatConductor] = None,
            chat_renderer: Optional[ChatRenderer] = None,
            max_total_messages: Optional[int] = None,
            hide_messages: bool = False,
            description: str = 'This chat room is a regular group chat. Everybody can talk to everybody else.'
    ):
        assert max_total_messages is None or max_total_messages > 0, 'Max total messages must be None or greater than 0.'

        self.participants = {}
        self.messages = sorted(initial_messages or [], key=lambda m: m.id)
        self.last_message_id = None if len(self.messages) == 0 else self.messages[-1].id
        self.chat_conductor = chat_conductor or BasicChatConductor()
        self.chat_renderer = chat_renderer or TerminalChatRenderer()
        self.hide_messages = hide_messages
        self.max_messages = max_total_messages
        self.description = description

        for i, participant in enumerate(initial_participants or []):
            self.add_participant(participant, should_elect_new_leader=i == 0)

    def add_participant(self, participant: ChatParticipant, should_elect_new_leader: bool = False):
        if participant.name in self.participants:
            raise ChatParticipantAlreadyJoinedToChat(participant.name)

        self.participants[participant.name] = participant

        if self.chat_leader is None:
            self.chat_leader = participant
        elif should_elect_new_leader:
            self.chat_leader = self.chat_conductor.elect_new_chat_leader(chat=self)

            for participant in self.participants.values():
                participant.on_new_chat_leader_elected(chat=self, new_leader=self.chat_leader)

        for participant in self.participants.values():
            participant.on_participant_joined_chat(chat=self, participant=participant)

    def remove_participant(self, participant: ChatParticipant):
        if participant.name not in self.participants:
            raise ChatParticipantNotJoinedToChat(participant.name)

        self.participants.pop(participant.name)

        for participant in self.participants.values():
            participant.on_participant_left_chat(chat=self, participant=participant)

        if self.chat_leader == participant:
            if len(self.participants) == 0:
                self.chat_leader = None
            else:
                self.chat_leader = self.chat_conductor.elect_new_chat_leader(chat=self)

            for participant in self.participants.values():
                participant.on_new_chat_leader_elected(chat=self, new_leader=self.chat_leader)

    def receive_message(self, sender_name: str, content: str):
        if sender_name not in self.participants:
            raise ChatParticipantNotJoinedToChat(sender_name)

        sender = self.participants[sender_name]

        self.last_message_id = self.last_message_id + 1 if self.last_message_id is not None else 1

        message = ChatMessage(
            id=self.last_message_id,
            sender_name=sender.name,
            content=content
        )

        self.messages.append(message)

        self.chat_renderer.render_new_chat_message(chat=self, message=message)

        for participant in self.participants.values():
            participant.on_new_chat_message(chat=self, message=message)

    def reset(self, remove_participants: bool = False):
        self.messages = []
        self.last_message_id = None

        if remove_participants:
            for participant in list(self.participants.values()):
                self.remove_participant(participant)

    def initiate_chat_with_result(
            self,
            initial_message: Optional[str] = None
    ) -> str:
        if len(self.participants) <= 1:
            raise NotEnoughParticipantsInChat(len(self.participants))

        if initial_message is not None:
            self.receive_message(sender_name=self.chat_leader.name, content=initial_message)

        next_speaker = self.chat_conductor.select_next_speaker(chat=self)
        while next_speaker is not None:
            message_content = next_speaker.respond_to_chat(chat=self)

            self.receive_message(sender_name=next_speaker.name, content=message_content)

            next_speaker = self.chat_conductor.select_next_speaker(chat=self)

        for participant in self.participants.values():
            participant.on_chat_ended(chat=self)

        result = self.chat_conductor.get_chat_result(chat=self)

        return result


class UserChatParticipant(ChatParticipant):
    def __init__(self, name: str = 'User', **kwargs):
        super().__init__(name, role='User', messages_hidden=True, **kwargs)

    def respond_to_chat(self, chat: 'ChatRoom') -> str:
        return input(f'👤 ({self.name}): ')


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

# RULES
- You can only send messages to other participants in the group chat. You cannot send messages to yourself ({name}).
- You do not have to respond directly to the one who sent you a message. You can respond to anyone in the group chat.

# INPUT
- Messages from the group chat, including your own messages.
  - They are prefixed by the sender's name (could also be everyone). For context only; it's not actually part of the message they sent. Example: "John: Hello, how are you?"
  - Some messages could have been sent by participants who are no longer a part of this conversation. Use their contents for context only; do not talk to them.

# OUTPUT
- Your response to a participant in the group chat.
- Do not prefix your message with your name. The system will do that for you.

# GOOD OUTPUT EXAMPLES
- "Hello, how are you?"
- "I am doing well, thanks. How are you?"

# BAD OUTPUT
- "John: Hello, how are you?"
- "Assistant: I am doing well, thanks. How are you?"
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
                 spinner: Optional[Halo] = None,
                 **kwargs
                 ):
        super().__init__(name=name, symbol=symbol, role=role, **kwargs)

        self.chat_model = chat_model
        self.spinner = spinner
        self.mission = mission

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

        return system_message

    def _chat_messages_to_chat_model_messages(self, chat_messages: List[ChatMessage]) -> List[BaseMessage]:
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

        system_message = self._create_system_message(chat=chat)

        all_messages = self._chat_messages_to_chat_model_messages(chat.messages)
        all_messages = [
            SystemMessage(content=system_message),
            *all_messages
        ]

        last_message = self.chat_model.predict_messages(all_messages)

        if self.spinner is not None:
            self.spinner.stop()

        message_content = last_message.content

        potential_prefix = f'{self.name}:'
        if message_content.startswith(potential_prefix):
            message_content = message_content[len(potential_prefix):].strip()

        return message_content


# class TeamBasedChatParticipant(ChatParticipant):
#     team_leader: ChatParticipant
#     other_team_participants: List[ChatParticipant]
#     team_interaction_schema: Optional[str] = None
#     hide_inner_chat: bool = True
#     max_sub_chat_messages: Optional[int] = None
#     team_chat_history_access: bool = True
#     spinner: Optional[Halo] = None
#
#     def __init__(self,
#                  team_leader: ChatParticipant,
#                  other_team_participants: List[ChatParticipant],
#                  team_interaction_schema: Optional[str] = None,
#                  hide_inner_chat: bool = True,
#                  max_total_sub_chat_messages: Optional[int] = None,
#                  team_chat_history_access: bool = True,
#                  spinner: Optional[Halo] = None,
#                  **kwargs):
#         super().__init__(name=team_leader.name, role=team_leader.role, **kwargs)
#
#         assert len(other_team_participants) >= 1, 'At least one team leader and 1 other team participant are required.'
#
#         self.other_team_participants = other_team_participants
#         self.team_leader = team_leader
#         self.hide_inner_chat = hide_inner_chat
#         self.max_sub_chat_messages = max_total_sub_chat_messages
#         self.team_chat_history_access = team_chat_history_access
#         self.team_interaction_schema = team_interaction_schema
#         self.spinner = spinner
#
#     def on_new_chat_message(self, chat: 'ChatRoom', messages: List['ChatMessage'], termination_received: bool = False):
#         if termination_received:
#             return
#
#         relevant_messages = [message for message in messages if self.name == message.recipient_name]
#         if len(relevant_messages) == 0:
#             return
#
#         last_message = relevant_messages[-1]
#
#         if self.spinner is not None:
#             self.spinner.stop_and_persist(symbol='👥',
#                                           text=f'{self.team_leader.name}\'s team started discussing the request.')
#             self.spinner.start(text=f'{self.team_leader.name}\'s team is actively discussing the request...')
#
#         sub_chat = ChatRoom(
#             initial_participants=[self.team_leader, *self.other_team_participants],
#             initial_messages=chat.messages if self.team_chat_history_access else None,
#             hide_messages=self.hide_inner_chat,
#             max_total_messages=self.max_sub_chat_messages,
#             chat_conductor=AIChatConductor()
#         )
#
#         response_from_team = sub_chat.initiate_chat_with_result(
#             first_message=f'{last_message.sender_name} has told me this: "{last_message.content}". What should I respond with?',
#             from_participant=self.team_leader,
#         )
#
#         self.send_message(chat=chat, content=response_from_team)
#
#         if self.spinner is not None:
#             self.spinner.succeed(text=f'{self.team_leader.name}\'s team discussion was concluded.')


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

    def respond_to_chat(self, chat: 'ChatRoom') -> str:
        if len(chat.messages) == 0:
            raise NoMessagesInChat()

        last_message = chat.messages[-1]

        try:
            json_string = last_message.content[last_message.content.index('{'):last_message.content.rindex('}') + 1]
            self.output = model = json_string_to_pydantic(json_string, self.output_schema)

            return f'{model.model_dump_json()} TERMINATE'
        except Exception as e:
            return f'I could not parse the JSON. This was the error: {e}'


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
        initial_participants=[json_parser, text_to_json_ai],
        hide_messages=hide_message,
        max_total_messages=n_retries * 2
    )

    _ = parser_chat.initiate_chat_with_result(
        initial_message=f'# TEXT\n{output}\n\n# JSON SCHEMA\n{pydantic_to_json_schema(output_schema)}'
    )

    if json_parser.output is None:
        raise MessageCouldNotBeParsed(output)

    return json_parser.output


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    chat_model = ChatOpenAI(temperature=0.0, model='gpt-4-0613')

    spinner = Halo(spinner='dots')

    ai = AIChatParticipant(name='Assistant', role='Boring AI Assistant', chat_model=chat_model, spinner=spinner)
    rob = AIChatParticipant(name='Rob', role='Funny Prankster',
                            mission='Collaborate with the user to prank the boring AI. Yawn.',
                            chat_model=chat_model, spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai, rob]

    main_chat = ChatRoom(
        initial_participants=participants,
        chat_conductor=AIChatConductor(
            chat_model=chat_model,
            speaker_interaction_schema=f'Rob should take the lead and go back and forth with the assistant trying to prank him big time. Rob can and should talk to the user to get them in on the prank, however the majority of the prank should be done by Rob. The chat should end when the AI is successfuly pranked once, when the user decides to end the chat, or if the AI is being too boring after 3 tries.',
            spinner=spinner
        ),
    )
    main_chat.initiate_chat_with_result()

    # ai = AIChatParticipant(name='AI', role='Math Expert',
    #                        mission='Solve the user\'s math problem and TERMINATE immediately (only if you have the solution). The user has only one problem.',
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
    #     output=main_chat.initiate_chat_with_result(
    #         first_message="Hey",
    #         from_participant=user,
    #         to_participant=ai
    #     ),
    #     chat_model=chat_model,
    #     output_schema=MathResult,
    #     spinner=spinner
    # )
    #
    # print(f'Result: {parsed_output}')
    #
    #
    # criteria_generation_team = TeamBasedChatParticipant(
    #     team_leader=AIChatParticipant(name='Tom',
    #                                   role='Criteria Generation Team Leader',
    #                                   mission=f'Delegate to your team and respond back with comprehensive, orthogonal, well-researched criteria for a decision-making problem. Respond as if you came up with the answer yourself. TERMINATE immediately when the research team gives you the answer.',
    #                                   chat_model=chat_model,
    #                                   spinner=spinner),
    #     other_team_participants=[
    #         AIChatParticipant(name='Rob',
    #                           role='Criteria Generator',
    #                           mission='Think from first principles about the decision-making problem, and come up with orthogonal, compresive list of criteria. Iterate on it, as needed.',
    #                           chat_model=chat_model,
    #                           spinner=spinner),
    #         AIChatParticipant(name='John',
    #                           role='Criteria Generation Critic',
    #                           mission='Collaborate with Rob to come up with a comprehensive, orthogonal list of criteria. Criticize Rob\'s criteria and provide counterfactual evidence to support your criticism. Iterate on it, as needed.',
    #                           chat_model=chat_model,
    #                           spinner=spinner),
    #     ],
    #     team_interaction_schema='The team leader will ask Rob to come up with a list of criteria. Rob will summarize the first principle approach to problem solving, and then come up with a list of initial criteria and give it John to criticize. Rob and John will go back and forth, refining and improving the criteria set until they both think the set cannot be improved anymore. Then, Rob will send the final set to the team leader, who will then send it back to the client.',
    #     hide_inner_chat=False,
    #     team_chat_history_access=False,
    #     spinner=spinner
    # )
    # user = UserChatParticipant(name='User')
    # participants = [user, criteria_generation_team]
    #
    # main_chat = ChatRoom(initial_participants=participants)
    # result = main_chat.initiate_chat_with_result(
    #     initial_message="Please generate a list of criteria for choosing the breed of my next puppy.",
    #     from_participant=user,
    #     to_participant=criteria_generation_team
    # ),

    # print(f'Result: {result}')
