import abc

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import TextSplitter
from pydantic import BaseModel

import research.prompts as system_prompts
from chat import chat
from research.page_retriever import PageRetriever
from bs4 import BeautifulSoup


def clean_html(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')

    # Remove all tag attributes
    for tag in soup.find_all(True):
        tag.attrs = {}

    # Remove any tag that doesn't have visible text
    for tag in soup.find_all(True):
        if not tag.get_text(strip=True):
            tag.extract()

    return str(soup.text)


class PageQueryAnalysisResult(BaseModel):
    answer: str
    context: str


class PageQueryAnalyzer(abc.ABC):
    @abc.abstractmethod
    def analyze(self, url: str, title: str, query: str) -> PageQueryAnalysisResult:
        raise NotImplementedError()


class OpenAIChatPageQueryAnalyzer(PageQueryAnalyzer):
    def __init__(self, chat_model: ChatOpenAI, page_retriever: PageRetriever, text_splitter: TextSplitter):
        self.chat_model = chat_model
        self.page_retriever = page_retriever
        self.text_splitter = text_splitter

    def analyze(self, url: str, title: str, query: str) -> PageQueryAnalysisResult:
        html = self.page_retriever.retrieve_html(url)
        html_text = clean_html(html)

        docs = self.text_splitter.create_documents([html_text])

        answer, context = 'No answer yet.', 'No context available.'
        for i, doc in enumerate(docs):
            text = doc.page_content
            result = chat(chat_model=self.chat_model, messages=[
                SystemMessage(content=system_prompts.answer_query_based_on_partial_page_system_prompt),
                HumanMessage(
                    content=f'# QUERY\n{query}\n\n# URL\n{url}\n\n# TITLE\n{title}\n\n# PREVIOUS ANSWER\n{answer}\n\n# CONTEXT\n{context}\n\n# PAGE TEXT\n{text}')
            ], result_schema=PageQueryAnalysisResult)

            answer, context = result.answer, result.context

        return PageQueryAnalysisResult(
            answer=answer,
            context=context
        )
