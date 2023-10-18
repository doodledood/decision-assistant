import re
from typing import Optional, Tuple

from halo import Halo
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from pydantic.v1 import BaseModel, Field

from chat import chat, json_string_to_pydantic
from research.page_analyzer import PageQueryAnalyzer
from research.search import SearchResultsProvider
import research.prompts as system_prompts


def url_contains_unsupported_file(url):
    # List of unsupported file types
    unsupported_types = ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'rtf', 'jpg', 'png', 'gif']

    # Extract file extension from the URL
    file_extension = re.findall(r'\.([a-zA-Z0-9]+)(?:[\?\#]|$)', url)

    # Check if the file extension is in the list of unsupported types
    if file_extension and file_extension[0] in unsupported_types:
        return True
    else:
        return False


class WebSearch:
    def __init__(self, chat_model: ChatOpenAI, search_results_provider: SearchResultsProvider,
                 page_query_analyzer: PageQueryAnalyzer):
        self.chat_model = chat_model
        self.search_results_provider = search_results_provider
        self.page_query_analyzer = page_query_analyzer

    def get_answer(self, query: str, n_results: int = 3, halo: Optional[Halo] = None) -> Tuple[bool, str]:
        search_results = self.search_results_provider.search(query=query, n_results=n_results)
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

        for result in search_results.organic_results:
            if url_contains_unsupported_file(result.link):
                continue

            if halo is not None:
                halo.text = f'Analyzing #{result.position} result "{result.title}"'

            page_result = self.page_query_analyzer.analyze(url=result.link, title=result.title, query=query)
            qna.append({
                'answer': page_result.answer,
                'source': result.link
            })

        if halo is not None:
            halo.succeed(f'Done searching the web.')
            halo.start(f'Processing results...')

        formatted_answers = '\n'.join([f'{i + 1}. {q["answer"]}; Source: {q["source"]}' for i, q in enumerate(qna)])
        final_answer = chat(chat_model=self.chat_model, messages=[
            SystemMessage(content=system_prompts.aggregate_query_answers_system_prompt),
            HumanMessage(content=f'# QUERY\n{query}\n\n# ANSWERS\n{formatted_answers}')
        ], max_ai_messages=1, use_halo=False, get_user_input=lambda x: 'terminate now please')

        return True, final_answer


class WebSearchToolArgs(BaseModel):
    query: str = Field(description='The query to search the web for.')


def create_web_search_tool(search: WebSearch, description: Optional[str] = None, n_results: int = 3):
    def get_answer(args: str):
        q: WebSearchToolArgs = json_string_to_pydantic(args, WebSearchToolArgs)

        with Halo(text='Searching the web...', spinner='dots') as spinner:
            return search.get_answer(query=q.query, n_results=n_results, halo=spinner)[1]

    web_search_tool = Tool.from_function(
        name='web_search',
        description=description if description else 'Search the web for information. Use this when the user requests information you don\'t know or if you are actively researching and need the most up to date data.',
        func=get_answer,
        args_schema=WebSearchToolArgs
    )

    return web_search_tool
