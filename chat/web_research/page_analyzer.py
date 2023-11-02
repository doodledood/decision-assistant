import abc
from typing import Optional

from halo import Halo
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TextSplitter
from pydantic import BaseModel

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor
from chat.parsing_utils import string_output_to_pydantic
from chat.participants import UserChatParticipant, LangChainBasedAIChatParticipant
from chat.renderers import NoChatRenderer
from chat.structured_string import Section, StructuredString
from chat.web_research.page_retriever import PageRetriever
from bs4 import BeautifulSoup


def extract_html_text(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')

    # Remove all tag attributes
    for tag in soup.find_all(True):
        tag.attrs = {}

    # Remove any tag that doesn't have visible text
    for tag in soup.find_all(True):
        if not tag.get_text(strip=True):
            tag.extract()

    text = soup.text

    # Close down double new lines to single new lines
    while '\n\n' in text:
        text = text.replace('\n\n', '\n')

    return text


class PageQueryAnalysisResult(BaseModel):
    answer: str


class PageQueryAnalyzer(abc.ABC):
    @abc.abstractmethod
    def analyze(self, url: str, title: str, query: str, spinner: Optional[Halo] = None) -> PageQueryAnalysisResult:
        raise NotImplementedError()


class OpenAIChatPageQueryAnalyzer(PageQueryAnalyzer):
    def __init__(self, chat_model: ChatOpenAI, page_retriever: PageRetriever, text_splitter: TextSplitter,
                 use_first_split_only: bool = True):
        self.chat_model = chat_model
        self.page_retriever = page_retriever
        self.text_splitter = text_splitter
        self.use_first_split_only = use_first_split_only

    def analyze(self, url: str, title: str, query: str, spinner: Optional[Halo] = None) -> PageQueryAnalysisResult:
        html = self.page_retriever.retrieve_html(url)
        html_text = extract_html_text(html)

        docs = self.text_splitter.create_documents([html_text])

        answer = 'No answer yet.'
        for i, doc in enumerate(docs):
            text = doc.page_content
            chat = Chat(
                backing_store=InMemoryChatDataBackingStore(),
                renderer=NoChatRenderer(),
                initial_participants=[
                    UserChatParticipant(),
                    LangChainBasedAIChatParticipant(
                        name='Web Page Query Answerer',
                        role='Web Page Query Answerer',
                        personal_mission='Answer queries based on provided (partial) web page data from the web.',
                        chat_model=self.chat_model,
                        other_prompt_sections=[
                            Section(name='Crafting a Query Answer', sub_sections=[
                                Section(name='Process', list=[
                                    'Analyze the query and the given data',
                                    'If context is provided, use it to answer the query.',
                                    'Summarize the answer in a comprehensive, yet succinct way.',
                                ], list_item_prefix=None),
                                Section(name='Guidelines', list=[
                                    'If the answer is not found in the page data, it\'s insufficent, or not relevant to the query at all, state it clearly.',
                                    'Do not fabricate information. Stick to provided data.',
                                    'Provide context for the next call (e.g., if a paragraph was cut short, include relevant header information, section, etc. for continuity). Assume the data is partial data from the page. Be very detailed in the context.',
                                    'If unable to answer but found important information, include it in the context for the next call.',
                                    'Pay attention to the details of the query and make sure the answer is suitable for the intent of the query.',
                                    'A potential answer might have been provided. This means you thought you found the answer in a previous partial text for the same page. You should double-check that and provide an alternative revised answer if you think it\'s wrong, or repeat it if you think it\'s right or cannot be validated using the current text.',
                                ])
                            ])
                        ]
                    )
                ], max_total_messages=2)
            chat_conductor = RoundRobinChatConductor()
            final_answer = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=str(
                StructuredString(sections=[
                    Section(name='Query', text=query),
                    Section(name='Url', text=url),
                    Section(name='Title', text=title),
                    Section(name='Previous Answer', text=answer),
                    Section(name='Page Text', text=text)
                ])
            ))
            result = string_output_to_pydantic(
                output=final_answer,
                chat_model=self.chat_model,
                output_schema=PageQueryAnalysisResult
            )
            answer = result.answer

            if self.use_first_split_only:
                break

        return PageQueryAnalysisResult(
            answer=answer,
        )
