from typing import Optional

from langchain.chat_models.openai import ChatOpenAI
from langchain.retrievers import WebResearchRetriever
from langchain.schema.vectorstore import VectorStore
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain


class WebResearch:
    def __init__(self, llm: ChatOpenAI, vectorstore: VectorStore, search: Optional[GoogleSearchAPIWrapper] = None):
        if search is None:
            search = GoogleSearchAPIWrapper()

        self.search = search
        self.vectorstore = vectorstore
        self.llm = llm

    def research_the_web(self, query: str):
        web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=self.vectorstore,
            llm=self.llm,
            search=self.search
        )
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(self.llm, retriever=web_research_retriever)

        return qa_chain({'question': query})
