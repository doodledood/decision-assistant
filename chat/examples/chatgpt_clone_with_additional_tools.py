import traceback
from typing import Type, Any, Optional

from halo import Halo
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.text_splitter import TokenTextSplitter
from langchain.tools import BaseTool
import pydantic.v1 as pydantic_v1

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

from chat.renderers import TerminalChatRenderer
from chat.web_research import WebSearch
from chat.web_research.page_analyzer import OpenAIChatPageQueryAnalyzer
from chat.web_research.page_retrievers.scraper_api_retriever import ScraperAPIPageRetriever
from chat.web_research.page_retrievers.selenium_retriever import SeleniumPageRetriever
from chat.web_research.search import GoogleSerperSearchResultsProvider
from chat.web_research.web_research import WebResearchTool


class CodeExecutionToolArgs(pydantic_v1.BaseModel):
    python_code: str = pydantic_v1.Field(
        description='The verbatim python code to execute. Ensure code always ends with `res = str(...)`, to capture the result of the code.')


class SimpleCodeExecutionTool(BaseTool):
    name: str = 'code_executor'
    description: str = 'Use this code executor for any capability you are missing expect for web searching. That includes math, time, data analysis, etc. Code will get executed and the result will be returned as a string.'
    args_schema: Type[pydantic_v1.BaseModel] = CodeExecutionToolArgs
    progress_text: str = 'Executing code...'
    spinner: Optional[Halo] = None

    def _run(
            self,
            python_code: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs: Any
    ) -> Any:
        if self.spinner is not None:
            self.spinner.stop_and_persist(symbol='🐍',
                                          text='Will execute the following code:\n```\n' + python_code + '\n```')
            self.spinner.start(self.progress_text)

        local_vars = {}

        try:
            exec(python_code, None, local_vars)
        except:
            return f'Error executing code: {traceback.format_exc()}'

        if 'res' not in local_vars:
            return 'Code did not include a `res = ...` statement, unable to read result. Retry again please.'

        res = local_vars['res']

        return res


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
            # Should `pip install selenium webdriver_manager` to use this
            page_retriever=SeleniumPageRetriever(),
            text_splitter=TokenTextSplitter(chunk_size=12000, chunk_overlap=2000),
            use_first_split_only=True
        )
    )

    spinner = Halo(spinner='dots')
    ai = LangChainBasedAIChatParticipant(
        name='Assistant',
        chat_model=chat_model,
        tools=[
            SimpleCodeExecutionTool(spinner=spinner),
            WebResearchTool(web_search=web_search, n_results=3, spinner=spinner)
        ],
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
