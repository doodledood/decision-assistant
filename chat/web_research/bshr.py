# Based directly on David Shaprio's BSHR Loop: https://github.com/daveshap/BSHR_Loop
import datetime
from typing import Callable, Optional

from dotenv import load_dotenv
from halo import Halo
from langchain.cache import SQLiteCache, GPTCache
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.globals import set_llm_cache

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from chat.renderers import TerminalChatRenderer
from chat.structured_prompt import Section, StructuredPrompt
from chat.web_research import WebSearch
from chat.web_research.page_analyzer import OpenAIChatPageQueryAnalyzer
from chat.web_research.page_retriever import ScraperAPIPageRetriever
from chat.web_research.search import GoogleSerperSearchResultsProvider
from chat.web_research.web_research import WebResearchTool

if __name__ == '__main__':
    load_dotenv()

    llm_cache = SQLiteCache()
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
    query_generator = LangChainBasedAIChatParticipant(
        name='Search Query Generator',
        role='Search Query Generator',
        personal_mission='You will be given a specific query or problem by the USER and you are to generate a JSON list of at most 5 questions that will be used to search the internet. Make sure you generate comprehensive and counterfactual search queries. Employ everything you know about information foraging and information literacy to generate the best possible questions.',
        other_prompt_sections=shared_sections,
        chat_model=chat_model,
        spinner=spinner)
    web_searcher = LangChainBasedAIChatParticipant(
        name='Web Searcher',
        role='Web Searcher',
        personal_mission='Search the web for all the queries generated in the last step by the query generator and give the answers found to each.',
        other_prompt_sections=shared_sections,
        tools=[
            WebResearchTool(web_search=web_search, n_results=3, spinner=spinner)
        ],
        chat_model=chat_model,
        spinner=spinner)
    hypothesis_generator = LangChainBasedAIChatParticipant(
        name='Information Needs Hypothesis Generator',
        role='Information Needs Hypothesis Generator',
        personal_mission='You are an information needs hypothesis generator. You will be given a main information need or user query as well as a variety of materials, such as search results, previous hypotheses, and notes. Whatever information you receive, your output should be a revised, refined, or improved hypothesis. In this case, the hypothesis is a comprehensive answer to the user query or information need. To the best of your ability. Do not include citations in your hypothesis, as this will all be record via out-of-band processes (e.g. the information that you are shown will have metadata and cataloging working behind the scenes that you do not see). Even so, you should endeavour to write everything in complete, comprehensive sentences and paragraphs such that your hypothesis requires little to no outside context to understand. Your hypothesis must be relevant to the USER QUERY or INFORMATION NEED.',
        other_prompt_sections=shared_sections,
        chat_model=chat_model,
        spinner=spinner)
    satisficing_checker = LangChainBasedAIChatParticipant(
        name='Information Needs Satisficing Checker',
        role='Information Needs Satisficing Checker',
        personal_mission='You are an information needs satisficing checker. You will be given a litany of materials, including an original user query, previous search queries, their results, notes, and a final hypothesis. You are to generate a decision as to whether or not the information need has been satisficed or not. You are to make this judgment by virtue of several factors: amount and quality of searches performed, specificity and comprehensiveness of the hypothesis, and notes about the information domain and foraging (if present). Several things to keep in mind: the user\'s information need may not be answerable, or only partially answerable, given the available information or nature of the problem.  Unanswerable data needs are satisficed when data foraging doesn\'t turn up more relevant information.',
        other_prompt_sections=shared_sections,
        chat_model=chat_model,
        spinner=spinner)
    spr_writer = LangChainBasedAIChatParticipant(
        name='SPR Writer',
        role='SPR Writer',
        personal_mission='You are a Sparse Priming Representation (SPR) writer. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation of Large Language Models (LLMs). You will be given information by the USER which you are to render as an SPR.',
        other_prompt_sections=shared_sections + [
            Section(
                name='Theory',
                text='LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of an LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.'
            ),
            Section(
                name='Methodology',
                text='Render the input as a distilled list of succinct statements, assertions, associations, concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as possible but with as few words as possible. Write it in a way that makes sense to you, as the future audience will be another language model, not a human.'
            )
        ],
        chat_model=chat_model,
        spinner=spinner)


    class UserWithEvidenceChatParticipant(UserChatParticipant):
        def __init__(self, get_evidence: Callable[[], Optional[str]], name: str = 'User'):
            super().__init__(name=name)

            self.get_evidence = get_evidence

        def on_new_chat_message(self, chat: 'Chat', message: 'ChatMessage'):
            user_response = super().on_new_chat_message(chat=chat, message=message)

            evidence = self.get_evidence()
            if evidence is None:
                return user_response

            return str(StructuredPrompt(
                sections=[
                    Section(name='Previous Evidence', text=evidence),
                    Section(name='User Comment', text=user_response)
                ]
            ))


    evidence = None

    user = UserWithEvidenceChatParticipant(name='User', get_evidence=lambda: evidence)
    participants = [user, hypothesis_generator, web_searcher, query_generator, satisficing_checker, spr_writer]

    while True:
        chat = Chat(
            goal='The goal of this chat is to generate a data-backed hypothesis that satisfices the information need of the user.',
            backing_store=InMemoryChatDataBackingStore(),
            renderer=TerminalChatRenderer(),
            initial_participants=participants
        )

        chat_conductor = RoundRobinChatConductor()
        evidence = chat_conductor.initiate_chat_with_result(chat=chat)

        print(evidence)
