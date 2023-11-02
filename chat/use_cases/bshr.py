# Based directly on David Shaprio's BSHR Loop: https://github.com/daveshap/BSHR_Loop
import datetime
import json
import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional, List, Dict, Set

from dotenv import load_dotenv
from halo import Halo
from langchain.cache import SQLiteCache, GPTCache
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.text_splitter import TokenTextSplitter
from langchain.globals import set_llm_cache
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor
from chat.parsing_utils import chat_messages_to_pydantic
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from chat.renderers import TerminalChatRenderer
from chat.structured_string import Section, StructuredString
from chat.web_research import WebSearch
from chat.web_research.page_analyzer import OpenAIChatPageQueryAnalyzer
from chat.web_research.page_retriever import ScraperAPIPageRetriever
from chat.web_research.search import GoogleSerperSearchResultsProvider
from chat.web_research.web_research import WebResearchTool
from sequential_process import SequentialProcess, Step


class BHSRState(BaseModel):
    information_need: Optional[str] = None
    queries_to_run: Optional[List[str]] = None
    answers_to_queries: Optional[Dict[str, str]] = None
    current_hypothesis: Optional[str] = None
    proposed_hypothesis: Optional[str] = None
    is_satisficed: Optional[bool] = None


def save_state(state: BHSRState, state_file: Optional[str]):
    if state_file is None:
        return

    data = state.model_dump()
    with open(state_file, 'w') as f:
        json.dump(data, f, indent=2)


class QueryGenerationResult(BaseModel):
    information_need: str = Field(description='Information need as requested by the user.')
    queries: List[str] = Field(description='Set of queries to run.')


class HypothesisGenerationResult(BaseModel):
    hypothesis: str = Field(description='A new or updated hypothesis based on the materials provided.')


class SatisficationCheckResult(BaseModel):
    is_satisficed: bool = Field(description='Whether or not the information need has been satisficed.')


def generate_queries(state: BHSRState,
                     chat_model: BaseChatModel,
                     shared_sections: Optional[List[Section]] = None,
                     web_search_tool: Optional[BaseTool] = None,
                     spinner: Optional[Halo] = None):
    query_generator = LangChainBasedAIChatParticipant(
        name='Search Query Generator',
        role='Search Query Generator',
        personal_mission='You will be given a specific query or problem by the USER and you are to generate a JSON list of at most 5 questions that will be used to search the internet. Make sure you generate comprehensive and counterfactual search queries. Employ everything you know about information foraging and information literacy to generate the best possible questions.',
        other_prompt_sections=shared_sections + [
            Section(name='Unclear Information Need',
                    text='If the information need or query are vague and unclear, either perform a web search to clarify the information need or ask the user for clarification'),
            Section(name='Refine Queries',
                    text='You might be given a first-pass information need, in which case you will do the best you can to generate "naive queries" (uninformed search queries). However the USER might also give you previous search queries or other background information such as accumulated notes. If these materials are present, you are to generate "informed queries" - more specific search queries that aim to zero in on the correct information domain. Do not duplicate previously asked questions. Use the notes and other information presented to create targeted queries and/or to cast a wider net.'),
            Section(name='Termination',
                    text='Once you generate a new set of queries to run, you should terminate the chat immediately by ending your message with TERMINATE')
        ],
        tools=[web_search_tool] if web_search_tool is not None else None,
        ignore_group_chat_environment=True,
        chat_model=chat_model,
        spinner=spinner)

    user = UserChatParticipant()
    participants = [user, query_generator]
    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()

    if state.information_need is None:
        if spinner is not None:
            spinner.stop()

        _ = chat_conductor.initiate_chat_with_result(
            chat=chat,
            initial_message=f'What is your information need or query?',
            from_participant=query_generator)
    else:
        _ = chat_conductor.initiate_chat_with_result(
            chat=chat,
            initial_message=str(StructuredString(sections=[
                Section(name='Information Need', text=state.information_need),
                Section(name='Previous Queries & Answers', sub_sections=[
                    Section(name=query, text=f'```markdown\n{answer}\n```', uppercase_name=False) for query, answer in
                    state.answers_to_queries.items()
                ]),
                Section(name='Current Hypothesis', text=state.current_hypothesis)
            ])),
            from_participant=user)

    output = chat_messages_to_pydantic(chat_messages=chat.get_messages(), chat_model=chat_model,
                                       output_schema=QueryGenerationResult)

    if state.information_need is None:
        state.information_need = output.information_need

    if state.queries_to_run is None:
        state.queries_to_run = []

    state.queries_to_run += output.queries


def search_queries(state: BHSRState,
                   web_search: WebSearch,
                   n_search_results: int = 3,
                   spinner: Optional[Halo] = None):
    if state.queries_to_run is None:
        return

    queries_and_answers = state.answers_to_queries if state.answers_to_queries is not None else {}
    queries_to_run_set = set(state.queries_to_run)
    for query in state.queries_to_run:
        if query in queries_and_answers:
            continue

        answer = web_search.get_answer(query=query, n_results=n_search_results, spinner=spinner)[1]

        queries_and_answers[query] = answer
        queries_to_run_set.remove(query)

        state.answers_to_queries = queries_and_answers
        state.queries_to_run = list(queries_to_run_set)

        yield state


def generate_hypothesis(state: BHSRState,
                        chat_model: BaseChatModel,
                        shared_sections: Optional[List[Section]] = None,
                        spinner: Optional[Halo] = None):
    user = UserChatParticipant()
    hypothesis_generator = LangChainBasedAIChatParticipant(
        name='Information Needs Hypothesis Generator',
        role='Information Needs Hypothesis Generator',
        personal_mission='You are an information needs hypothesis generator. You will be given a main information need or user query as well as a variety of materials, such as search results, previous hypotheses, and notes. Whatever information you receive, your output should be a revised, refined, or improved hypothesis. In this case, the hypothesis is a comprehensive answer to the user query or information need. To the best of your ability. Do not include citations in your hypothesis, as this will all be record via out-of-band processes (e.g. the information that you are shown will have metadata and cataloging working behind the scenes that you do not see). Even so, you should endeavour to write everything in complete, comprehensive sentences and paragraphs such that your hypothesis requires little to no outside context to understand. Your hypothesis must be relevant to the USER QUERY or INFORMATION NEED.',
        other_prompt_sections=shared_sections + [
            Section(name='Termination',
                    text='Once you generate a new or updated hypothesis, you should terminate the chat immediately by ending your message with TERMINATE')
        ],
        ignore_group_chat_environment=True,
        chat_model=chat_model,
        spinner=spinner)
    participants = [user, hypothesis_generator]
    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()

    _ = chat_conductor.initiate_chat_with_result(
        chat=chat,
        initial_message=str(StructuredString(sections=[
            Section(name='Information Need', text=state.information_need),
            Section(name='Previous Queries & Answers', sub_sections=[
                Section(name=query, text=f'```markdown\n{answer}\n```', uppercase_name=False) for query, answer in
                state.answers_to_queries.items()
            ]),
            Section(name='Previous Hypothesis', text=state.current_hypothesis)
        ])),
        from_participant=user)

    output = chat_messages_to_pydantic(chat_messages=chat.get_messages(), chat_model=chat_model,
                                       output_schema=HypothesisGenerationResult)
    state.proposed_hypothesis = output.hypothesis


def check_satisficing(state: BHSRState,
                      chat_model: BaseChatModel,
                      shared_sections: Optional[List[Section]] = None,
                      spinner: Optional[Halo] = None):
    user = UserChatParticipant()
    satisficing_checker = LangChainBasedAIChatParticipant(
        name='Information Needs Satisficing Checker',
        role='Information Needs Satisficing Checker',
        personal_mission='You are an information needs satisficing checker. You will be given a litany of materials, including an original user query, previous search queries, their results, notes, and a final hypothesis. You are to generate a decision as to whether or not the information need has been satisficed or not. You are to make this judgment by virtue of several factors: amount and quality of searches performed, specificity and comprehensiveness of the hypothesis, and notes about the information domain and foraging (if present). Several things to keep in mind: the user\'s information need may not be answerable, or only partially answerable, given the available information or nature of the problem.  Unanswerable data needs are satisficed when data foraging doesn\'t turn up more relevant information. Use a step-by-step approach to determine whether or not the information need has been satisficed.',
        other_prompt_sections=shared_sections + [
        Section(name='Termination',
                text='Once you determine if the information need has been satisfied or not, you should terminate the chat immediately by ending your message with TERMINATE')
        ],
        ignore_group_chat_environment=True,
        chat_model=chat_model,
        spinner=spinner)
    participants = [user, satisficing_checker]
    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()

    _ = chat_conductor.initiate_chat_with_result(
        chat=chat,
        initial_message=str(StructuredString(sections=[
            Section(name='Information Need', text=state.information_need),
            Section(name='Previous Queries & Answers', sub_sections=[
                Section(name=query, text=f'```markdown\n{answer}\n```', uppercase_name=False) for query, answer in
                state.answers_to_queries.items()
            ]),
            Section(name='Previous Hypothesis', text=state.current_hypothesis),
            Section(name='Proposed New Hypothesis', text=state.proposed_hypothesis)
        ])),
        from_participant=user)

    output = chat_messages_to_pydantic(chat_messages=chat.get_messages(), chat_model=chat_model,
                                       output_schema=SatisficationCheckResult)

    state.is_satisficed = output.is_satisficed
    state.current_hypothesis = state.proposed_hypothesis
    state.proposed_hypothesis = None


if __name__ == '__main__':
    load_dotenv()

    output_dir = Path(os.getenv('OUTPUT_DIR', '../../output'))
    output_dir.mkdir(exist_ok=True, parents=True)

    n_search_results = 2

    state_file = str(output_dir / 'bshr_state.json')
    llm_cache = SQLiteCache(database_path=str(output_dir / 'llm_cache.db'))
    set_llm_cache(llm_cache)

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
                model='gpt-3.5-turbo-16k-0613',
            ),
            page_retriever=ScraperAPIPageRetriever(),
            text_splitter=TokenTextSplitter(chunk_size=12000, chunk_overlap=2000),
            use_first_split_only=True
        )
    )

    spinner = Halo(spinner='dots')
    shared_sections = [
        Section(
            name='Current Date (YYYY-MM-DD)',
            text=datetime.datetime.utcnow().strftime('%Y-%m-%d')
        )
    ]
    web_search_tool = WebResearchTool(web_search=web_search, n_results=n_search_results, spinner=spinner)

    process = SequentialProcess(
        steps=[
            Step(
                name='Query Generation',
                func=partial(
                    generate_queries,
                    chat_model=chat_model,
                    shared_sections=shared_sections,
                    web_search_tool=web_search_tool,
                    spinner=spinner
                ),
                on_step_start=lambda _: spinner.start('Generating queries...'),
                on_step_completed=lambda _: spinner.succeed('Queries generated.')
            ),
            Step(
                name='Web Search',
                func=partial(
                    search_queries,
                    web_search=web_search,
                    n_search_results=n_search_results,
                    spinner=spinner
                ),
                on_step_start=lambda _: spinner.start('Searching queries...'),
                on_step_completed=lambda _: spinner.succeed('Queries answered.')
            ),
            Step(
                name='Hypothesis Generation',
                func=partial(
                    generate_hypothesis,
                    chat_model=chat_model,
                    shared_sections=shared_sections,
                    spinner=spinner
                ),
                on_step_start=lambda _: spinner.start('Generating hypothesis...'),
                on_step_completed=lambda _: spinner.succeed('Hypothesis generated.')
            ),
            Step(
                name='Satificing Check',
                func=partial(
                    check_satisficing,
                    chat_model=chat_model,
                    shared_sections=shared_sections,
                    spinner=spinner
                ),
                on_step_start=lambda _: spinner.start('Checking satisfication condition...'),
                on_step_completed=lambda _: spinner.succeed('Satisfication checked.')
            ),
        ],
        initial_state=BHSRState(),
        save_state=partial(save_state, state_file=state_file)
    )

    while True:
        state = process.run()
        if state.is_satisficed:
            break

    print(f'Final Answer:\n============\n\n{state.current_hypothesis}')