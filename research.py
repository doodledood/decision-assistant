from typing import Optional

from langchain.chat_models.openai import ChatOpenAI
from langchain.retrievers import WebResearchRetriever
from langchain.schema.vectorstore import VectorStore
from langchain.tools import Tool
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain


class WebSearch:
    def __init__(self, llm: ChatOpenAI, vectorstore: VectorStore, search: Optional[GoogleSearchAPIWrapper] = None):
        if search is None:
            search = GoogleSearchAPIWrapper()

        self.search = search
        self.vectorstore = vectorstore
        self.llm = llm

    def get_answer(self, query: str, n_result_pages: int = 1) -> str:
        web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=self.vectorstore,
            llm=self.llm,
            search=self.search,
            num_search_results=n_result_pages
        )
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(self.llm, retriever=web_research_retriever)

        return str(qa_chain({'question': query}))


def create_web_search_tool(search: WebSearch, description: Optional[str] = None):
    web_search_tool = Tool(
        name='web_search',
        description=description if description else 'Search the web for information. Use this when the user requests information you don\'t know or if you are actively researching and need the most up to date data.',
        func=search.get_answer
    )

    return web_search_tool
