from typing import Dict, Any, Optional, List

from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
from langchain.tools import BaseTool

from chat.ai_utils import execute_chat_model_messages
from chat.base import ChatCompositionGenerator, Chat, GeneratedChatComposition
from chat.composition_generators import ManageParticipantsOutputSchema
from chat.parsing_utils import string_output_to_pydantic
from chat.participants import LangChainBasedAIChatParticipant
from chat.structured_prompt import StructuredPrompt, Section


class LangChainBasedAIChatCompositionGenerator(ChatCompositionGenerator):
    chat_model: BaseChatModel
    chat_model_args: Dict[str, Any]
    tools: Optional[List[BaseTool]] = None,
    spinner: Optional[Halo] = None
    n_output_parsing_tries: int = 3

    def __init__(self,
                 chat_model: BaseChatModel,
                 tools: Optional[List[BaseTool]] = None,
                 chat_model_args: Optional[Dict[str, Any]] = None,
                 spinner: Optional[Halo] = None,
                 n_output_parsing_tries: int = 3):
        self.chat_model = chat_model
        self.chat_model_args = chat_model_args or {}
        self.tools = tools
        self.spinner = spinner
        self.n_output_parsing_tries = n_output_parsing_tries

    def generate_composition_for_chat(self, chat: Chat) -> GeneratedChatComposition:
        if self.spinner is not None:
            self.spinner.start(text='The Chat Composition Generator is creating a new chat composition...')

        # Ask the AI to select the next speaker.
        messages = [
            SystemMessage(content=self.create_compose_chat_participants_system_prompt(chat=chat)),
            HumanMessage(content=self.create_compose_chat_participants_first_human_prompt(chat=chat))
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

        if self.spinner is not None:
            if len(output.participants_to_remove) == 0 and len(output.participants_to_add) == 0:
                self.spinner.succeed(
                    text='The Chat Composition Generator has decided to keep the current chat composition.')
            elif len(output.participants_to_remove) > 0 and len(output.participants_to_add) == 0:
                self.spinner.succeed(
                    text=f'The Chat Composition Generator has decided to remove the following participants: '
                         f'{", ".join(output.participants_to_remove)}')
            elif len(output.participants_to_remove) == 0 and len(output.participants_to_add) > 0:
                self.spinner.succeed(
                    text=f'The Chat Composition Generator has decided to add the following participants: '
                         f'{", ".join([participant.name for participant in output.participants_to_add])}')
            else:
                self.spinner.succeed(
                    text=f'The Chat Composition Generator has decided to remove the following participants: '
                         f'{", ".join(output.participants_to_remove)} and add the following participants: '
                         f'{", ".join([participant.name for participant in output.participants_to_add])}')

        participants = [p for p in chat.get_active_participants() if p.name not in output.participants_to_remove]

        for participant in output.participants_to_add:
            participants.append(LangChainBasedAIChatParticipant(
                name=participant.name,
                role=participant.role,
                personal_mission=participant.personal_mission,
                symbol=participant.symbol,
                chat_model=self.chat_model,
                tools=self.tools,
                spinner=self.spinner,
                chat_model_args=self.chat_model_args
            ))

        return GeneratedChatComposition(
            participants=participants,
            participants_interaction_schema=output.updated_speaker_interaction_schema
        )

    def create_compose_chat_participants_system_prompt(self, chat: 'Chat') -> str:
        active_participants = chat.get_active_participants()
        system_message = StructuredPrompt(sections=[
            Section(name='Mission',
                    text='Evaluate the chat conversation based on the set goal and the speaker interaction schema. Make decisions about adding or removing participants based on their potential contribution towards achieving the goal. Update the interaction schema to reflect changes in participants.'),
            Section(name='Process', list=[
                'Think about the ideal composition of participants that can contribute to the goal in a step-by-step manner by looking at all the inputs.',
                'Assess if the current participants are sufficient for ideally contributing to the goal.',
                'If insufficient, summon additional participants as needed.',
                'If some participants are unnecessary, remove them.',
                'Update the interaction schema to accommodate changes in participants.'
            ], list_item_prefix=None),
            Section(name='Adding Participants', list=[
                'Add participants based on their potential contribution to the goal.',
                'Generate a name, role, and personal mission for each new participant such that they can contribute to the goal the best they can, each in their complementary own way.',
                'Always try to add or complete comprehensive teams of competent specialist participants that have orthogonal and complementary skills/roles.',
                'Since most participants you summon will not be the best experts in the world, even though they think they are, they will be to be overseen. For that, most tasks will require at least 2 experts, one doing a task and the other that will act as a critic to that expert; they can loop back and forth and iterate on a better answer. For example, instead of having a Planner only, have a Planner and a Plan Critic participants to have this synergy. You can skip critics for the most trivial tasks.',
                'You may not necessarily have the option to change this composition later, so make sure you summon the right participants.',
            ]),
            Section(name='Removing Participants', list=[
                'Remove participants only if they cannot contribute to the goal or fit into the interaction schema.',
                'Ignore past performance. Focus on the participant\'s potential contribution to the goal and their fit into the interaction schema.'
            ]),
            Section(name='Updating The Speaker Interaction Schema',
                    list=[
                        'Update the interaction schema to accommodate changes in participants.',
                        'The interaction schema should provide guidelines for a chat manager on how to coordinate the participants to achieve the goal. Like an algorithm for choosing the next speaker in the conversation.',
                        'The goal of the chat (if provided) must be included in the interaction schema. The whole purpose of the interaction schema is to help achieve the goal.',
                        'It should be very clear how the process goes and when it should end.',
                        'The interaction schema should be simple, concise, and very focused on the goal. Formalities should be avoided, unless they are necessary for achieving the goal.',
                        'If the chat goal has some output (like an answer), make sure to have the last step be the presentation of the final answer by one of the participants as a final message to the chat.'
                    ]),
            Section(name='Input', list=[
                'Goal for the conversation.',
                'Previous messages from the conversation.',
                'Speaker interaction schema.'
            ]),
            Section(name='Output',
                    text='The output can be compressed, as it will not be used by a human, but by an AI. It should include:',
                    list=[
                        'Participants to Remove: List of participants to be removed (if any).',
                        'Participants to Add: List of participants to be added, with their name, role, and personal mission.',
                        'Updated Interaction Schema: An updated version of the original interaction schema.'
                    ])
        ])

        return str(system_message)

    def create_compose_chat_participants_first_human_prompt(self, chat: Chat) -> str:
        messages = chat.get_messages()
        messages_list = [f'- {message.sender_name}: {message.content}' for message in messages]

        active_participants = chat.get_active_participants()

        prompt = StructuredPrompt(sections=[
            Section(name='Chat Goal', text=chat.goal or 'No explicit chat goal provided.'),
            Section(name='Currently Active Participants',
                    list=[f'{participant.name} ({participant.role})' for participant in active_participants]),
            Section(name='Current Speaker Interaction Schema',
                    text=chat.speaker_interaction_schema or 'Not provided. Use your best judgement.'),
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
