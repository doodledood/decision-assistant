from functools import partial
from typing import Optional

from dotenv import load_dotenv
from fire import Fire
from halo import Halo
from langchain.callbacks import StreamingStdOutCallbackHandler, StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter

from research import create_web_search_tool, WebSearch
from research.page_analyzer import OpenAIChatPageQueryAnalyzer
from research.page_retriever import ScraperAPIPageRetriever
from research.search import GoogleSerperSearchResultsProvider
from sequential_process import Step, SequentialProcess
from state import DecisionAssistantState, load_state, save_state
from steps import identify_goal, identify_alternatives, identify_criteria, map_criteria, prioritize_criteria, \
    generate_research_questions, research_data, analyze_data, compile_data_for_presentation, present_report


def run_decision_assistant(
        goal: Optional[str] = None,
        llm_temperature: float = 0.0,
        llm_model: str = 'gpt-4-0613',
        fast_llm_model: str = 'gpt-3.5-turbo-16k-0613',
        state_file: Optional[str] = 'state.json',
        report_file: str = 'decision_report.html',
        streaming: bool = False,
        n_search_results: int = 3,
        render_js_when_researching: bool = False
):
    spinner = Halo(spinner='dots')

    chat_model = ChatOpenAI(temperature=llm_temperature, model=llm_model, streaming=streaming,
                            callbacks=[StreamingStdOutCallbackHandler() if streaming else StdOutCallbackHandler()])
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
    default_tools_with_web_search = [
        create_web_search_tool(search=web_search, n_results=n_search_results, spinner=spinner)]

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
                func=partial(identify_goal, chat_model, default_tools_with_web_search, spinner=spinner),
                on_step_start=lambda _: spinner.start('Identifying goal...'),
                on_step_completed=lambda _: spinner.succeed('Identified goal.')
            ),
            Step(
                name='Alternative Listing',
                func=partial(identify_alternatives, chat_model, default_tools_with_web_search, spinner=spinner),
                on_step_start=lambda _: spinner.start('Identifying alternatives...'),
                on_step_completed=lambda _: spinner.succeed('Identified alternatives.')
            ),
            Step(
                name='Criteria Identification',
                func=partial(identify_criteria, chat_model, default_tools_with_web_search, spinner=spinner),
                on_step_start=lambda _: spinner.start('Identifying criteria...'),
                on_step_completed=lambda _: spinner.succeed('Identified criteria.')
            ),
            Step(
                name='Criteria Mapping',
                func=partial(map_criteria, chat_model, default_tools_with_web_search, spinner=spinner),
                on_step_start=lambda _: spinner.start('Mapping criteria...'),
                on_step_completed=lambda _: spinner.succeed('Mapped criteria.')
            ),
            Step(
                name='Criteria Prioritization',
                func=prioritize_criteria,
                on_step_start=lambda _: spinner.start('Prioritizing criteria...'),
                on_step_completed=lambda _: spinner.succeed('Prioritized criteria.')
            ),
            Step(
                name='Research Questions Generation',
                func=partial(generate_research_questions, chat_model, default_tools_with_web_search, spinner=spinner),
                on_step_start=lambda _: spinner.start('Generating research questions...'),
                on_step_completed=lambda _: spinner.succeed('Generated research questions.')
            ),
            Step(
                name='Data Research',
                func=partial(research_data, chat_model, web_search, n_search_results, default_tools_with_web_search,
                             spinner=spinner),
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
