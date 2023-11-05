from halo import Halo
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.backing_stores.langchain import LangchainMemoryBasedChatDataBackingStore
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
        model='gpt-4-0613'
    )

    spinner = Halo(spinner='dots')
    ai = LangChainBasedAIChatParticipant(
        name='Assistant',
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai]

    memory = ConversationSummaryBufferMemory(llm=chat_model, max_token_limit=100)
    chat = Chat(
        backing_store=LangchainMemoryBasedChatDataBackingStore(memory=memory),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    chat_conductor.initiate_chat_with_result(chat=chat)
