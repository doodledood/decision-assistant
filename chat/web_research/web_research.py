import re
from typing import Optional, Tuple

from halo import Halo
from langchain.chat_models.openai import ChatOpenAI
from pydantic import BaseModel, Field

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from chat.renderers import NoChatRenderer
from chat.web_research.page_analyzer import PageQueryAnalyzer
from chat.web_research.search import SearchResultsProvider
from chat.structured_prompt import Section, StructuredPrompt

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

                page_result = self.page_query_analyzer.analyze(url=result.link, title=result.title, query=query,
                                                               spinner=spinner)

                if spinner is not None:
                    spinner.succeed(f'Read & analyzed #{result.position} result "{result.title}".')

                qna.append({
                    'answer': page_result.answer,
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
                                'If the answer is not found in the page data, state it clearly.',
                                'Should be formatted in Markdown with inline citations.'
                            ])
                        ])
                    ]
                )
            ], max_total_messages=2)
        chat_conductor = RoundRobinChatConductor()
        final_answer = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=str(
            StructuredPrompt(sections=[
                Section(name='Query', text=query),
                Section(name='Answers', text=formatted_answers)
            ])
        ))

        if spinner is not None:
            spinner.succeed(f'Done searching the web.')

        return True, final_answer


class SearchTheWeb(BaseModel):
    """Search the web. Use that to get an answer for a query you don't know the answer to, for recent events, or if the user asks you to."""
    query: str = Field(description='The query to search the web for.')


def answer_query(web_search: WebSearch, args: SearchTheWeb, n_search_results: int = 3, spinner: Optional[Halo] = None):
    return web_search.get_answer(query=args.query, n_results=n_search_results, spinner=spinner)[1]
