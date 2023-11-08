from halo import Halo
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationSummaryBufferMemory

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.backing_stores.langchain import LangChainMemoryBasedChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

from chat.participants.langchain import LangChainBasedAIChatParticipant
from chat.participants.user import UserChatParticipant
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

    try:
        memory = ConversationSummaryBufferMemory(
            llm=chat_model,
            max_token_limit=OpenAI.modelname_to_contextsize(chat_model.model_name)
        )
        backing_store = LangChainMemoryBasedChatDataBackingStore(memory=memory)
    except ValueError:
        backing_store = InMemoryChatDataBackingStore()

    chat = Chat(
        backing_store=backing_store,
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    chat_conductor.initiate_chat_with_result(chat=chat)
