from typing import Dict, Any, Optional, List, Callable

from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
from langchain.tools import BaseTool

from chat.ai_utils import execute_chat_model_messages
from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import ChatCompositionGenerator, Chat, GeneratedChatComposition, ChatConductor
from chat.composition_generators import ManageParticipantsOutputSchema
from chat.conductors import LangChainBasedAIChatConductor
from chat.parsing_utils import string_output_to_pydantic
from chat.participants.internal_group import InternalGroupBasedChatParticipant
from chat.participants.langchain import LangChainBasedAIChatParticipant
from chat.renderers import TerminalChatRenderer
from chat.structured_string import StructuredString, Section


class LangChainBasedAIChatCompositionGenerator(ChatCompositionGenerator):
    chat_model: BaseChatModel
    chat_model_args: Dict[str, Any]
    tools: Optional[List[BaseTool]] = None,
    spinner: Optional[Halo] = None
    n_output_parsing_tries: int = 3
    prefer_critics: bool = False

    def __init__(self,
                 chat_model: BaseChatModel,
                 tools: Optional[List[BaseTool]] = None,
                 chat_model_args: Optional[Dict[str, Any]] = None,
                 spinner: Optional[Halo] = None,
                 n_output_parsing_tries: int = 3,
                 prefer_critics: bool = False):
        self.chat_model = chat_model
        self.chat_model_args = chat_model_args or {}
        self.tools = tools
        self.spinner = spinner
        self.n_output_parsing_tries = n_output_parsing_tries
        self.prefer_critics = prefer_critics

    def generate_composition_for_chat(self,
                                      chat: Chat,
                                      composition_suggestion: Optional[str] = None,
                                      participants_interaction_schema: Optional[str] = None,
                                      termination_condition: Optional[str] = None,
                                      create_internal_chat: Optional[
                                          Callable[[str], Chat]] = None) -> GeneratedChatComposition:
        if create_internal_chat is None:
            create_internal_chat = lambda goal: Chat(
                goal=goal,
                backing_store=InMemoryChatDataBackingStore(),
                renderer=TerminalChatRenderer()
            )

        if self.spinner is not None:
            self.spinner.start(text='The Chat Composition Generator is creating a new chat composition...')

        # Ask the AI to select the next speaker.
        messages = [
            SystemMessage(content=self.create_compose_chat_participants_system_prompt(chat=chat)),
            HumanMessage(
                content=self.create_compose_chat_participants_first_human_prompt(
                    chat=chat,
                    composition_suggestion=composition_suggestion,
                    participants_interaction_schema=participants_interaction_schema,
                    termination_condition=termination_condition))
        ]

        result = self.execute_messages(messages=messages)

        output = string_output_to_pydantic(
            output=result,
            chat_model=self.chat_model,
            output_schema=ManageParticipantsOutputSchema,
            n_tries=self.n_output_parsing_tries,
            spinner=self.spinner,
            hide_message=False
        )

        participants_to_add_names = [str(participant) for participant in output.participants_to_add]
        participants_to_remove_names = [participant_name for participant_name in output.participants_to_remove]

        if self.spinner is not None:
            if len(output.participants_to_remove) == 0 and len(output.participants_to_add) == 0:
                self.spinner.succeed(
                    text='The Chat Composition Generator has decided to keep the current chat composition.')
            elif len(output.participants_to_remove) > 0 and len(output.participants_to_add) == 0:
                self.spinner.succeed(
                    text=f'The Chat Composition Generator has decided to remove the following participants: '
                         f'{", ".join(participants_to_remove_names)}')
            elif len(output.participants_to_remove) == 0 and len(output.participants_to_add) > 0:
                self.spinner.succeed(
                    text=f'The Chat Composition Generator has decided to add the following participants: '
                         f'{", ".join(participants_to_add_names)}')
            else:
                self.spinner.succeed(
                    text=f'The Chat Composition Generator has decided to remove the following participants: '
                         f'{", ".join(participants_to_remove_names)} and add the following participants: '
                         f'{", ".join(participants_to_add_names)}')

        participants = [p for p in chat.get_active_participants() if p.name not in output.participants_to_remove]

        for participant in output.participants_to_add:
            if participant.type == 'individual':
                chat_participant = LangChainBasedAIChatParticipant(
                    name=participant.name,
                    role=participant.role,
                    personal_mission=participant.mission,
                    symbol=participant.symbol,
                    chat_model=self.chat_model,
                    tools=self.tools,
                    spinner=self.spinner,
                    chat_model_args=self.chat_model_args
                )
            else:
                chat_participant = InternalGroupBasedChatParticipant(
                    group_name=participant.name,
                    mission=participant.mission,
                    chat=create_internal_chat(participant.mission),
                    chat_conductor=LangChainBasedAIChatConductor(
                        chat_model=self.chat_model,
                        chat_model_args=self.chat_model_args,
                        spinner=self.spinner,
                        composition_generator=LangChainBasedAIChatCompositionGenerator(
                            chat_model=self.chat_model,
                            tools=self.tools,
                            chat_model_args=self.chat_model_args,
                            spinner=self.spinner,
                            n_output_parsing_tries=self.n_output_parsing_tries
                        )
                    ),
                    spinner=self.spinner
                )

            participants.append(chat_participant)

        return GeneratedChatComposition(
            participants=participants,
            participants_interaction_schema=output.updated_speaker_interaction_schema,
            termination_condition=output.updated_termination_condition
        )

    def create_compose_chat_participants_system_prompt(self, chat: 'Chat') -> str:
        active_participants = chat.get_active_participants()

        adding_participants = [
            'Add participants based on their potential contribution to the goal.',
            'Generate a name, role, and personal mission for each new participant such that they can contribute '
            'to the goal the best they can, each in their complementary own way.',
            'Always try to add or complete comprehensive teams of competent specialist participants that have '
            'orthogonal and complementary skills, roles, and missions. You can also add teams instead of individual '
            'participants for a hierarchical structure when complexity needs to be reduced.',
            'Adding a team means interacting with their lead participant, who will be the front face of the team (do '
            'not worry about who that is though; by adding a team you harness the power of the entire team).'
            'Roles for individuals should be succinct titles like "Writer", "Developer", etc.'
            'You may not necessarily have the option to change this composition later, so make sure you summon '
            'the right participants.'
        ]

        if self.prefer_critics:
            adding_participants.append(
                'Since most participants you summon will not be the best experts in the world, even though they think '
                'they are, they will be to be overseen. For that, most tasks will require at least 2 experts, '
                'one doing a task and the other that will act as a critic to that expert; they can loop back and '
                'forth and iterate on a better answer. For example, instead of having a Planner only, have a Planner '
                'and a Plan Critic participants to have this synergy. You can skip critics for the most trivial tasks.'
            )

        system_message = StructuredString(sections=[
            Section(name='Mission',
                    text='Evaluate the chat conversation based on the INPUT. '
                         'Make decisions about adding or removing participants based on their potential contribution '
                         'towards achieving the goal. Update the interaction schema and the termination condition '
                         'to reflect changes in participants.'),
            Section(name='Process', list=[
                'Think about the ideal composition of participants that can contribute to the goal in a step-by-step '
                'manner by looking at all the inputs.',
                'Assess if the current participants are sufficient for ideally contributing to the goal.',
                'If insufficient, summon additional participants as needed.',
                'If some participants are unnecessary, remove them.',
                'Update the interaction schema to accommodate changes in participants.'
            ], list_item_prefix=None),
            Section(name='Participants Composition', sub_sections=[
                Section(name='Adding Participants', list=adding_participants, sub_sections=[
                    Section(name='Team-based Participants', list=[
                        'For very difficult tasks, you may need to summon a team of participants to work together to '
                        'achieve the goal.',
                        'You can summon a team the same way you do individual participants but include a team when '
                        'describing them.',
                        'This team attribute will be used to summon an entire group of internal participants that will '
                        'make up the team. Do not worry about the team\'s composition at this point.',
                        'By specifying a team parameter, you are saying that the participant is the front face of the '
                        'team and their representative in the chat. If you do not specify a team, the participant will '
                        'be a solo specialist participant.',
                        'When summoning a team, role is not important, you can just leave that blank or use the name of'
                        'the team as the role. Assume an entire sub-team will be in place of a participant and will '
                        'have their own separate group chat whenever they need to respond.'
                    ])
                ]),
                Section(name='Removing Participants', list=[
                    'Remove participants only if they cannot contribute to the goal or fit into the interaction schema.',
                    'Ignore past performance. Focus on the participant\'s potential contribution to the goal and their '
                    'fit into the interaction schema.'
                ]),
                Section(name='Order of Participants', list=[
                    'The order of participants is important. It should be the order in which they should be summoned.',
                    'The first participant will be regarded to as the leader of the group at times, so make sure to '
                    'choose the right one to put first.',
                ]),
                Section(name='Orthogonality of Participants', list=[
                    'Always strive to have participants with orthogonal skills and roles. That includes personal '
                    'missions, as well.',
                    'Shared skills and missions is a waste of resources.'
                ]),
                Section(name='Composition Suggestion',
                        text='If provided by the user, take into account the '
                             'composition suggestion when deciding how to add/remove.'),
            ]),
            Section(name='Updating The Speaker Interaction Schema',
                    list=[
                        'Update the interaction schema to accommodate changes in participants.',
                        'The interaction schema should provide guidelines for a chat manager on how to coordinate the '
                        'participants to achieve the goal. Like an algorithm for choosing the next speaker in the '
                        'conversation.',
                        'The goal of the chat (if provided) must be included in the interaction schema. The whole '
                        'purpose of the interaction schema is to help achieve the goal.',
                        'It should be very clear how the process goes and when it should end.',
                        'The interaction schema should be simple, concise, and very focused on the goal. Formalities '
                        'should be avoided, unless they are necessary for achieving the goal.',
                        'If the chat goal has some output (like an answer), make sure to have the last step be the '
                        'presentation of the final answer by one of the participants as a final message to the chat.'
                    ]),
            Section(name='Updating The Termination Condition',
                    list=[
                        'Update the termination condition to accommodate changes in participants.',
                        'The termination condition should be a simple, concise, and very focused on the goal.',
                        'The chat should terminate when the goal is achieved, or when it is clear that the goal '
                        'cannot be achieved.'
                    ]),
            Section(name='Input', list=[
                'Goal for the conversation',
                'Previous messages from the conversation',
                'Current speaker interaction schema (Optional)',
                'Current termination condition (Optional)',
                'Composition suggestion (Optional)'
            ]),
            Section(name='Output',
                    text='The output can be compressed, as it will not be used by a human, but by an AI. It should '
                         'include:',
                    list=[
                        'Participants to Remove: List of participants to be removed (if any).',
                        'Participants to Add: List of participants to be added, with their name, role, team (if '
                        'applicable), and personal (or team) mission.',
                        'Updated Interaction Schema: An updated version of the original interaction schema.',
                        'Updated Termination Condition: An updated version of the original termination condition.'
                    ])
        ])

        return str(system_message)

    def create_compose_chat_participants_first_human_prompt(self, chat: Chat,
                                                            composition_suggestion: Optional[str] = None,
                                                            participants_interaction_schema: Optional[str] = None,
                                                            termination_condition: Optional[str] = None) -> str:
        messages = chat.get_messages()
        messages_list = [f'- {message.sender_name}: {message.content}' for message in messages]

        active_participants = chat.get_active_participants()

        prompt = StructuredString(sections=[
            Section(name='Chat Goal', text=chat.goal or 'No explicit chat goal provided.'),
            Section(name='Currently Active Participants',
                    list=[str(participant) for participant in active_participants]),
            Section(name='Current Speaker Interaction Schema',
                    text=participants_interaction_schema or 'Not provided. Use your best judgement.'),
            Section(name='Current Termination Condition',
                    text=termination_condition or 'Not provided. Use your best judgement.'),
            Section(name='Composition Suggestion',
                    text=composition_suggestion or 'Not provided. Use your best judgement.'),
            Section(name='Chat Messages',
                    text='No messages yet.' if len(messages_list) == 0 else None,
                    list=messages_list if len(messages_list) > 0 else []
                    )
        ])

        return str(prompt)

    def execute_messages(self, messages: List[BaseMessage]) -> str:
        return execute_chat_model_messages(
            messages=messages,
            chat_model=self.chat_model,
            tools=self.tools,
            spinner=self.spinner,
            chat_model_args=self.chat_model_args
        )
