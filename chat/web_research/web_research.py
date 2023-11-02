import re
from typing import Optional, Tuple, Type, Any

from halo import Halo
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models.openai import ChatOpenAI
from langchain.tools import Tool, BaseTool
from pydantic.v1 import BaseModel, Field
from tenacity import RetryError

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from chat.renderers import NoChatRenderer
from chat.web_research.errors import TransientHTTPError, NonTransientHTTPError
from chat.web_research.page_analyzer import PageQueryAnalyzer
from chat.web_research.search import SearchResultsProvider
from chat.structured_string import Section, StructuredString

video_watch_urls_patterns = [
    r'youtube.com/watch\?v=([a-zA-Z0-9_-]+)',
    r'youtu.be/([a-zA-Z0-9_-]+)',
    r'vimeo.com/([0-9]+)',
    r'dailymotion.com/video/([a-zA-Z0-9]+)',
    r'dailymotion.com/embed/video/([a-zA-Z0-9]+)',
    r'tiktok.com/@([a-zA-Z0-9_]+)/video/([0-9]+)'
]


def url_unsupported(url):
    # List of unsupported file types
    unsupported_types = ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'rtf', 'jpg', 'png', 'gif']

    # Extract file extension from the URL
    file_extension = re.findall(r'\.([a-zA-Z0-9]+)(?:[\?\#]|$)', url)

    # Check if the file extension is in the list of unsupported types
    if file_extension and file_extension[0] in unsupported_types:
        return True

    # Check if URL is a video or video site
    for pattern in video_watch_urls_patterns:
        if re.search(pattern, url):
            return True

    return False


class WebSearch:
    def __init__(self, chat_model: ChatOpenAI, search_results_provider: SearchResultsProvider,
                 page_query_analyzer: PageQueryAnalyzer, skip_results_if_answer_snippet_found: bool = True):
        self.chat_model = chat_model
        self.search_results_provider = search_results_provider
        self.page_query_analyzer = page_query_analyzer
        self.skip_results_if_answer_snippet_found = skip_results_if_answer_snippet_found

    def get_answer(self, query: str, n_results: int = 3, spinner: Optional[Halo] = None) -> Tuple[bool, str]:
        if spinner is not None:
            spinner.start(f'Getting search results for "{query}"...')

        search_results = self.search_results_provider.search(query=query, n_results=n_results)

        if spinner is not None:
            spinner.succeed(f'Got search results for "{query}".')

        if len(search_results.organic_results) == 0 and search_results.answer_snippet is None:
            return False, 'Nothing was found on the web for this query.'

        qna = []

        if search_results.knowledge_graph_description is not None:
            qna.append({
                'answer': search_results.knowledge_graph_description,
                'source': 'Knowledge Graph'
            })

        if search_results.answer_snippet is not None:
            qna.append({
                'answer': search_results.answer_snippet,
                'source': 'Answer Snippet'
            })

        if not self.skip_results_if_answer_snippet_found or search_results.answer_snippet is None:
            for result in search_results.organic_results:
                if url_unsupported(result.link):
                    continue

                if spinner is not None:
                    spinner.start(f'Reading & analyzing #{result.position} result "{result.title}"')

                try:
                    page_result = self.page_query_analyzer.analyze(url=result.link, title=result.title, query=query,
                                                                   spinner=spinner)
                    answer = page_result.answer

                    if spinner is not None:
                        spinner.succeed(f'Read & analyzed #{result.position} result "{result.title}".')
                except Exception as e:
                    if type(e) in (RetryError, TransientHTTPError, NonTransientHTTPError):
                        if spinner is not None:
                            spinner.warn(
                                f'Failed to read & analyze #{result.position} result "{result.title}", moving on.')

                        answer = 'Unable to answer query because the page could not be read.'
                    else:
                        raise

                qna.append({
                    'answer': answer,
                    'source': result.link
                })

        if spinner is not None:
            spinner.start(f'Processing results...')

        formatted_answers = '\n'.join([f'{i + 1}. {q["answer"]}; Source: {q["source"]}' for i, q in enumerate(qna)])

        chat = Chat(
            backing_store=InMemoryChatDataBackingStore(),
            renderer=NoChatRenderer(),
            initial_participants=[
                UserChatParticipant(),
                LangChainBasedAIChatParticipant(
                    name='Query Answer Aggregator',
                    role='Query Answer Aggregator',
                    personal_mission='Analyze query answers, discard unlikely ones, and provide an aggregated final response.',
                    chat_model=self.chat_model,
                    other_prompt_sections=[
                        Section(name='Aggregating Query Answers', sub_sections=[
                            Section(name='Process', list=[
                                'Receive query and answers with sources.',
                                'Analyze answers, discard unlikely or minority ones.',
                                'Formulate final answer based on most likely answers.',
                                'If no data found, respond "The answer could not be found."',
                            ], list_item_prefix=None),
                            Section(name='Aggregation', list=[
                                'Base final answer on sources.',
                                'Incorporate sources as inline citations in Markdown format.',
                                'Example: "Person 1 was [elected president in 2012](https://...)."',
                                'Only include sources from provided answers.',
                                'If part of an answer is used, use the same links inline.',
                            ]),
                            Section(name='Final Answer Notes', list=[
                                'Do not fabricate information. Stick to provided data.',
                                'You will be given the top search results from a search engine, there is a reason they are the top results. You should pay attention to all of them and think about the query intent.'
                                'If the answer is not found in the page data, state it clearly.',
                                'Should be formatted in Markdown with inline citations.'
                            ])
                        ])
                    ]
                )
            ], max_total_messages=2)
        chat_conductor = RoundRobinChatConductor()
        final_answer = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=str(
            StructuredString(sections=[
                Section(name='Query', text=query),
                Section(name='Answers', text=formatted_answers)
            ])
        ))

        if spinner is not None:
            spinner.succeed(f'Done searching the web.')

        return True, final_answer


class WebSearchToolArgs(BaseModel):
    query: str = Field(description='The query to search the web for.')


class WebResearchTool(BaseTool):
    web_search: WebSearch
    n_results: int = 3
    spinner: Optional[Halo] = None
    name: str = 'web_search'
    description: str = "Research the web. Use that to get an answer for a query you don't know or unsure of the answer to, for recent events, or if the user asks you to. This will evaluate answer snippets, knowledge graphs, and the top N results from google and aggregate a result."
    args_schema: Type[BaseModel] = WebSearchToolArgs
    progress_text: str = 'Searching the web...'

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs: Any
    ) -> Any:
        return self.web_search.get_answer(query=query, n_results=self.n_results, spinner=self.spinner)[1]
