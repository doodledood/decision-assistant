from halo import Halo
from langchain.text_splitter import TokenTextSplitter
from pydantic import BaseModel, Field

from chat.ai_utils import pydantic_to_openai_function
from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

from chat.renderers import TerminalChatRenderer
from research import WebSearch
from research.page_analyzer import OpenAIChatPageQueryAnalyzer
from research.page_retriever import ScraperAPIPageRetriever
from research.search import GoogleSerperSearchResultsProvider


class SearchTheWeb(BaseModel):
    """Searches the web for a given query that is inferred by the AI based on the conversation."""

    query: str = Field(description='The query to search for on the web.')


if __name__ == '__main__':
    load_dotenv()
    chat_model = ChatOpenAI(
        temperature=0.0,
        model='gpt-4-0613'
    )

    web_search = WebSearch(
        chat_model=chat_model,
        search_results_provider=GoogleSerperSearchResultsProvider(),
        page_query_analyzer=OpenAIChatPageQueryAnalyzer(
            chat_model=ChatOpenAI(
                temperature=0.0,
                model='gpt-3.5-turbo-16k-0613'
            ),
            page_retriever=ScraperAPIPageRetriever(),
            text_splitter=TokenTextSplitter(chunk_size=12000, chunk_overlap=2000),
            use_first_split_only=True
        )
    )

    spinner = Halo(spinner='dots')
    ai = LangChainBasedAIChatParticipant(
        name='Assistant',
        chat_model=chat_model,
        chat_model_args={
            'functions': [pydantic_to_openai_function(SearchTheWeb)]
        },
        functions={
            SearchTheWeb.__name__: lambda query: web_search.get_answer(query=query, n_results=3)[1]
        },
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
