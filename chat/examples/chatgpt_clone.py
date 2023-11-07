from halo import Halo

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

from chat.renderers import TerminalChatRenderer

if __name__ == '__main__':
    load_dotenv()
    chat_model = ChatOpenAI(
        temperature=0.0,
        model='gpt-4-1106-preview'
    )

    spinner = Halo(spinner='dots')
    ai = LangChainBasedAIChatParticipant(
        name='Assistant',
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    chat_conductor.initiate_chat_with_result(chat=chat)
