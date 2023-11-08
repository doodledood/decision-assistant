from halo import Halo

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.composition_generators.langchain import LangChainBasedAIChatCompositionGenerator
from chat.conductors import LangChainBasedAIChatConductor
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

from chat.participants.user import UserChatParticipant
from chat.renderers import TerminalChatRenderer

if __name__ == '__main__':
    load_dotenv()
    chat_model = ChatOpenAI(
        temperature=0.0,
        model='gpt-4-1106-preview'
    )

    spinner = Halo(spinner='dots')
    user = UserChatParticipant(name='User')
    chat_conductor = LangChainBasedAIChatConductor(
        chat_model=chat_model,
        spinner=spinner,
        # Pass in a composition generator to the conductor
        composition_generator=LangChainBasedAIChatCompositionGenerator(
            chat_model=chat_model,
            spinner=spinner,
        )
    )
    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        # Set up a proper goal so the composition generator can use it to generate the composition that will best fit
        goal='Come up with a plan for the user to invest their money. The goal is to maximize wealth over the '
             'long-term, while minimizing risk.',
        initial_participants=[user],
    )

    result = chat_conductor.initiate_chat_with_result(chat=chat)
    print(result)