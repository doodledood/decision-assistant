from functools import partial
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fire import Fire
from halo import Halo
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter

from chat.web_research import WebSearch
from chat.web_research.page_analyzer import OpenAIChatPageQueryAnalyzer
from chat.web_research.page_retriever import ScraperAPIPageRetriever
from chat.web_research.search import GoogleSerperSearchResultsProvider
from chat.web_research.web_research import WebResearchTool
from sequential_process import Step, SequentialProcess
from state import DecisionAssistantState, load_state, save_state
from steps import identify_goal, identify_alternatives, identify_criteria, map_criteria, prioritize_criteria, \
    generate_research_questions, perform_research, analyze_data, compile_data_for_presentation, present_report


def run_decision_assistant(
        goal: Optional[str] = None,
        llm_temperature: float = 0.0,
        llm_model: str = 'gpt-4-0613',
        fast_llm_model: str = 'gpt-3.5-turbo-16k-0613',
        state_file: Optional[str] = 'output/state.json',
        report_file: str = 'output/decision_report.html',
        n_search_results: int = 3,
        render_js_when_researching: bool = False,
        fully_autonomous_research: bool = True
):
    if state_file is not None:
        Path(state_file).parent.mkdir(exist_ok=True, parents=True)

    if report_file is not None:
        Path(report_file).parent.mkdir(exist_ok=True, parents=True)

    spinner = Halo(spinner='dots')

    chat_model = ChatOpenAI(temperature=llm_temperature, model=llm_model)
    fast_chat_model = ChatOpenAI(temperature=llm_temperature, model=fast_llm_model)
    web_search = WebSearch(
        chat_model=chat_model,
        search_results_provider=GoogleSerperSearchResultsProvider(),
        page_query_analyzer=OpenAIChatPageQueryAnalyzer(
            chat_model=fast_chat_model,
            page_retriever=ScraperAPIPageRetriever(render_js=render_js_when_researching),
            text_splitter=TokenTextSplitter(chunk_size=12000, chunk_overlap=2000)
        )
    )
    default_participant_tools = [
        WebResearchTool(web_search=web_search, n_results=n_search_results, spinner=spinner)
    ]

    spinner.start('Loading previous state...')
    state = load_state(state_file)
    if state is None:
        state = DecisionAssistantState(data={})
        spinner.stop()
    else:
        spinner.succeed('Loaded previous state.')

    process = SequentialProcess(
        steps=[
            Step(
                name='Goal Identification',
                func=partial(identify_goal, chat_model, tools=default_participant_tools, spinner=spinner),
                on_step_start=lambda _: spinner.start('Identifying goal...'),
                on_step_completed=lambda _: spinner.succeed('Identified goal.')
            ),
            Step(
                name='Alternative Listing',
                func=partial(identify_alternatives, chat_model, default_participant_tools, spinner=spinner),
                on_step_start=lambda _: spinner.start('Identifying alternatives...'),
                on_step_completed=lambda _: spinner.succeed('Identified alternatives.')
            ),
            Step(
                name='Criteria Identification',
                func=partial(identify_criteria, chat_model, default_participant_tools, spinner=spinner),
                on_step_start=lambda _: spinner.start('Identifying criteria...'),
                on_step_completed=lambda _: spinner.succeed('Identified criteria.')
            ),
            Step(
                name='Criteria Mapping',
                func=partial(map_criteria, chat_model, default_participant_tools, spinner=spinner),
                on_step_start=lambda _: spinner.start('Mapping criteria...'),
                on_step_completed=lambda _: spinner.succeed('Mapped criteria.')
            ),
            Step(
                name='Criteria Prioritization',
                func=prioritize_criteria,
                on_step_start=lambda _: spinner.succeed('Started prioritizing criteria.'),
                on_step_completed=lambda _: spinner.succeed('Prioritized criteria.')
            ),
            Step(
                name='Research Questions Generation',
                func=partial(generate_research_questions, chat_model, default_participant_tools, spinner=spinner),
                on_step_start=lambda _: spinner.start('Generating research questions...'),
                on_step_completed=lambda _: spinner.succeed('Generated research questions.')
            ),
            Step(
                name='Data Research',
                func=partial(perform_research, chat_model, web_search, n_search_results, default_participant_tools,
                             spinner=spinner, fully_autonomous=fully_autonomous_research),
                on_step_start=lambda _: spinner.start('Researching data...'),
                on_step_completed=lambda _: spinner.succeed('Researched data.')
            ),
            Step(
                name='Data Analysis',
                func=analyze_data,
                on_step_start=lambda _: spinner.start('Analyzing data...'),
                on_step_completed=lambda _: spinner.succeed('Analyzed data.')
            ),
            Step(
                name='Report Generation',
                func=partial(compile_data_for_presentation, report_file=report_file),
                on_step_start=lambda _: spinner.start('Producing report...'),
                on_step_completed=lambda _: spinner.succeed('Produced report.')
            ),
            Step(
                name='Report Presentation',
                func=partial(present_report, report_file=report_file),
                on_step_start=lambda _: spinner.start('Presenting report...'),
                on_step_completed=lambda _: spinner.succeed('Presented report.')
            )
        ],
        initial_state=state,
        save_state=partial(save_state, state_file=state_file)
    )

    process.run()


if __name__ == '__main__':
    load_dotenv()

    Fire(run_decision_assistant)
