from halo import Halo

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.composition_generators.langchain import LangChainBasedAIChatCompositionGenerator
from chat.conductors import LangChainBasedAIChatConductor, RoundRobinChatConductor
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

from chat.participants.internal_group import InternalGroupBasedChatParticipant
from chat.participants.user import UserChatParticipant
from chat.renderers import TerminalChatRenderer

if __name__ == '__main__':
    load_dotenv()
    chat_model = ChatOpenAI(
        temperature=0.0,
        model='gpt-4-1106-preview'
    )

    spinner = Halo(spinner='dots')
    comedy_team = InternalGroupBasedChatParticipant(
        group_name='Financial Team',
        chat=Chat(
            backing_store=InMemoryChatDataBackingStore(),
            renderer=TerminalChatRenderer(),
            goal='Ensure the user\'s financial strategy maximizes wealth over the long term without too much risk.',
            composition_generator=LangChainBasedAIChatCompositionGenerator(
                chat_model=chat_model,
                spinner=spinner,
            )
        ),
        chat_conductor=LangChainBasedAIChatConductor(
            chat_model=chat_model,
            spinner=spinner,
            termination_condition='Terminate when the team has come up with good solutions and suggestions for '
                                  'improvement of the plan or request of the user. Also terminate if user input is '
                                  'needed before a proper answer can be crafted.'
        ),
        spinner=spinner
    )
    user = UserChatParticipant(name='User')
    participants = [user, comedy_team]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()

    chat_conductor.initiate_chat_with_result(chat=chat)
