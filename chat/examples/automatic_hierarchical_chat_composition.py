from halo import Halo
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

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

    set_llm_cache(SQLiteCache(database_path='../../output/llm_cache.db'))

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
        goal='The goal is to create the best website for the user.',
        initial_participants=[user],
    )

    # Not necessary in practice since initiation is done automatically when calling `initiate_chat_with_result`.
    # However, this is needed to eagerly generate the composition. Default is lazy.
    chat_conductor.initialize_chat(
        chat=chat,
        # Only relevant when passing in a composition generator
        composition_suggestion='DevCompany: Includes a CEO, Product Team, Marketing Team, and a Development '
                               'Department. The Development Department includes a Director, QA Team and Development '
                               'Team.')
    print(f'\nGenerated composition:\n=================\n{chat.active_participants_str}\n=================\n\n')

    # You can also pass in a composition suggestion here.
    result = chat_conductor.initiate_chat_with_result(chat=chat)
    print(result)
