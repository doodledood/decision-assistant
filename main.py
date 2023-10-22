import itertools
import json
from typing import Optional, List, Tuple, Dict, Callable

import ahpy
import questionary
from dotenv import load_dotenv
from fire import Fire
from halo import Halo
from langchain.callbacks import StreamingStdOutCallbackHandler, StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import TokenTextSplitter
from pydantic.v1 import BaseModel, Field

import system_prompts
from chat import chat
from presentation import open_html_file_in_browser, generate_decision_report_as_html, save_html_to_file
from research import create_web_search_tool, WebSearch
from research.page_analyzer import OpenAIChatPageQueryAnalyzer
from research.page_retriever import ScraperAPIPageRetriever
from research.ranking import topsis_score, normalize_label_value
from research.search import GoogleSerperSearchResultsProvider
from state import Stage, DecisionAssistantState, load_state, save_and_mark_stage_as_done, mark_stage_as_done, save_state


class Criterion(BaseModel):
    name: str = Field(description='The name of the criterion. Example: "Affordability".')
    scale: List[str] = Field(
        description='The scale of the criterion, from worst to best. Labels only. No numerical value, no explainations. Example: "Very Expensive".')


class CriteriaIdentificationResult(BaseModel):
    criteria: List[Criterion] = Field(description='The identified criteria for evaluating the decision.')


class AlternativeListingResult(BaseModel):
    alternatives: List[str] = Field(description='The identified alternatives for the decision.')


class CriterionMappingResult(BaseModel):
    criterion_mapping: str = Field(
        description='An explaination for the criterion on how to assign a value from the scale to a piece of data.')


class CriteriaPrioritizationResult(BaseModel):
    criteria_weights: List[int] = Field(
        description='The weights of the criteria, from 1 to 100, reflecting their relative importance based on the user\'s preferences. Ordered in the same way as the criteria.')


class CriteriaResearchQueriesResult(BaseModel):
    criteria_research_queries: List[List[str]] = Field(
        description='The research queries for each criteria. Ordered in the same way as the criteria.')


class AlternativeCriteriaResearchFindingsResult(BaseModel):
    updated_research_findings: str = Field(
        description='The updated and aggregated research findings for the alternative and criterion. Formatted as rich markdown with all the citations and links in place.')
    label: str = Field(
        description='The label assigned to the alternative and criterion based on the aggregated research findings and user discussion. The label is assigned from the scale of the criterion (name of the label).')


class Alternative(BaseModel):
    name: str = Field(description='The name of the alternative.')
    criteria_data: Optional[Dict[str, Tuple[str, int]]] = Field(
        description='The research data collected for each criterion for this alternative. Key is the name of the criterion. Value is a tuple of the research data as text and the assigned value based on the scale of the criterion.')


def gather_unique_pairwise_comparisons(criteria_names: List[str],
                                       previous_comparisons: Optional[Dict[Tuple[str, str], float]] = None,
                                       on_question_asked: Optional[Callable[[Tuple[str, str], float], None]] = None) \
        -> Dict[Tuple[str, str], float]:
    choices = {
        'Much less important': 1 / 9,
        'Slightly less important': 1 / 5,
        'Equally important': 1,
        'Slightly more important': 5,
        'Much more important': 9
    }
    ordered_choice_names = [choice[0] for choice in sorted(choices.items(), key=lambda x: x[1])]

    comparisons = previous_comparisons
    all_combs = list(itertools.combinations(criteria_names, 2))
    for i, (label1, label2) in enumerate(all_combs):
        if (label1, label2) in comparisons:
            continue

        answer = questionary.select(
            f'({i + 1}/{len(all_combs)}) How much more important is "{label1}" when compared to "{label2}"?',
            choices=ordered_choice_names,
            default=ordered_choice_names[2]
        ).ask()

        labels = (label1, label2)
        value = choices[answer]

        comparisons[labels] = value

        if on_question_asked is not None:
            on_question_asked(labels, value)

    return comparisons


def run_decision_assistant(goal: Optional[str] = None, llm_temperature: float = 0.0, llm_model: str = 'gpt-4-0613',
                           fast_llm_model: str = 'gpt-3.5-turbo-16k-0613',
                           state_file: Optional[str] = 'state.json',
                           report_file: str = 'decision_report.html',
                           streaming: bool = False,
                           n_search_results: int = 3, render_js_when_researching: bool = False):
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
        state = DecisionAssistantState(stage=None, data={})
        spinner.stop()
    else:
        spinner.succeed('Loaded previous state.')

    # Identify goal
    if state.last_completed_stage is None and goal is None:
        spinner.start('Identifying goal...')

        goal = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.goal_identification_system_prompt),
            HumanMessage(content="Hey")
        ], tools=default_tools_with_web_search, spinner=spinner)

        state.last_completed_stage = Stage.GOAL_IDENTIFICATION
        state.data = {**state.data, **dict(goal=goal)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.GOAL_IDENTIFICATION)

    goal = state.data['goal']

    # Identify alternatives
    if state.last_completed_stage == Stage.GOAL_IDENTIFICATION:
        spinner.start('Identifying alternatives...')

        alternatives = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.alternative_listing_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}'),
        ], tools=default_tools_with_web_search, result_schema=AlternativeListingResult, spinner=spinner)
        alternatives = alternatives.dict()['alternatives']

        state.last_completed_stage = Stage.ALTERNATIVE_LISTING
        state.data = {**state.data, **dict(alternatives=alternatives)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.ALTERNATIVE_LISTING)

    alternatives = state.data['alternatives']

    # Identify criteria
    if state.last_completed_stage == Stage.ALTERNATIVE_LISTING:
        spinner.start('Identifying criteria...')

        criteria = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_identification_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}'),
        ], tools=default_tools_with_web_search, result_schema=CriteriaIdentificationResult, spinner=spinner)
        criteria = criteria.dict()['criteria']

        state.last_completed_stage = Stage.CRITERIA_IDENTIFICATION
        state.data = {**state.data, **dict(criteria=criteria)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_IDENTIFICATION)

    criteria = state.data['criteria']
    criteria_names = [criterion['name'] for criterion in criteria]

    # Map criteria
    if state.last_completed_stage == Stage.CRITERIA_IDENTIFICATION:
        spinner.start('Mapping criteria...')

        criteria_mapping = state.data.get('criteria_mapping', {})

        for criterion in criteria:
            if criterion['name'] in criteria_mapping:
                continue

            scale_str = '\n'.join([f'{i + 1}. {scale_value}' for i, scale_value in enumerate(criterion['scale'])])
            criterion_mapping = chat(chat_model=chat_model, messages=[
                SystemMessage(content=system_prompts.criterion_mapping_system_prompt),
                HumanMessage(
                    content=f'# GOAL\n{goal}\n\n# CRITERION NAME\n{criterion["name"]}\n\n# CRITERION SCALE\n{scale_str}'),
            ], tools=default_tools_with_web_search, result_schema=CriterionMappingResult, spinner=spinner)
            criterion_mapping = criterion_mapping.dict()['criterion_mapping']
            criteria_mapping[criterion['name']] = criterion_mapping

            state.data = {**state.data,
                          **dict(criteria_mapping=criteria_mapping)}
            save_state(state, state_file)

        state.last_completed_stage = Stage.CRITERIA_MAPPING
        state.data = {**state.data, **dict(criteria_mapping=criteria_mapping)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_MAPPING)

    criteria_mapping = state.data['criteria_mapping']

    # Prioritize criteria
    if state.last_completed_stage == Stage.CRITERIA_MAPPING:
        # spinner.start('Prioritizing criteria...')

        criteria_comparisons = state.data.get('criteria_comparisons', {})
        criteria_comparisons = {tuple(json.loads(labels)): value for labels, value in criteria_comparisons.items()}

        def save_comparison(labels, value):
            criteria_comparisons[labels] = value

            state.data = {**state.data, **dict(
                criteria_comparisons={json.dumps(labels): value for labels, value in criteria_comparisons.items()})}
            save_state(state, state_file)

        criteria_comparisons = gather_unique_pairwise_comparisons(criteria_names,
                                                                  previous_comparisons=criteria_comparisons,
                                                                  on_question_asked=save_comparison)
        criteria_weights = ahpy.Compare('Criteria', criteria_comparisons).target_weights
        criteria_weights = [int(round(criteria_weights[criterion_name] * 100)) for criterion_name in criteria_names]

        state.last_completed_stage = Stage.CRITERIA_PRIORITIZATION
        state.data = {**state.data, **dict(criteria_weights=criteria_weights)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_PRIORITIZATION)

    criteria_weights = state.data['criteria_weights']

    # Generate research questions
    if state.last_completed_stage == Stage.CRITERIA_PRIORITIZATION:
        spinner.start('Generating research questions...')

        criteria_research_queries = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_research_questions_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}\n\n# CRITERIA MAPPING\n{criteria_mapping}'),
        ], tools=default_tools_with_web_search, result_schema=CriteriaResearchQueriesResult,
                                         max_ai_messages=1,
                                         get_user_input=lambda x: 'terminate now please', spinner=spinner)
        criteria_research_queries = criteria_research_queries.dict()['criteria_research_queries']

        state.last_completed_stage = Stage.CRITERIA_RESEARCH_QUESTIONS_GENERATION
        state.data = {**state.data, **dict(criteria_research_queries=criteria_research_queries)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_RESEARCH_QUESTIONS_GENERATION)

    criteria_research_queries = state.data['criteria_research_queries']

    # Research data
    if state.last_completed_stage == Stage.CRITERIA_RESEARCH_QUESTIONS_GENERATION:
        spinner.start('Researching data...')

        research_data = state.data.get('research_data')
        if research_data is None:
            research_data = {}

        for alternative in alternatives:
            alternative_research_data = research_data.get(alternative)

            if alternative_research_data is None:
                alternative_research_data = {}

            for i, (criterion, criterion_research_questions) in enumerate(zip(criteria, criteria_research_queries)):
                criterion_name = criterion['name']
                alternative_criterion_research_data = alternative_research_data.get(criterion_name)

                if alternative_criterion_research_data is None:
                    alternative_criterion_research_data = {'raw': {}, 'aggregated': {}}

                # Already researched and aggregated, skip
                if alternative_criterion_research_data['aggregated'] != {}:
                    continue

                # Research data online for each query
                for query in criterion_research_questions:
                    query = query.format(alternative=alternative)

                    # Already researched query, skip
                    if query in alternative_criterion_research_data['raw']:
                        continue

                    found_answer, answer = web_search.get_answer(query=query, n_results=n_search_results,
                                                                 spinner=spinner)

                    if not found_answer:
                        alternative_criterion_research_data['raw'][query] = 'No answer found online.'

                        spinner.warn(f'No answer found for query "{query}".')
                    else:
                        alternative_criterion_research_data['raw'][query] = answer

                    alternative_research_data[criterion_name] = alternative_criterion_research_data
                    research_data[alternative] = alternative_research_data
                    state.data['research_data'] = research_data

                    save_state(state, state_file)

                # Present research data, discuss, aggregate, assign a proper label, and confirm with the user
                criterion_mapping = criteria_mapping[criterion_name]
                criterion_full_research_data = chat(chat_model=chat_model, messages=[
                    SystemMessage(content=system_prompts.alternative_criteria_research_system_prompt),
                    HumanMessage(
                        content=f'# GOAL\n{goal}\n\n# ALTERNATIVE\n{alternative}\n\n# CRITERION MAPPING\n{criterion_mapping}\n\n# RESEARCH FINDINGS\n{alternative_criterion_research_data}'),
                ], tools=default_tools_with_web_search, result_schema=AlternativeCriteriaResearchFindingsResult,
                                                    spinner=spinner)

                research_data[alternative][criterion_name]['aggregated'] = {
                    'findings': criterion_full_research_data.updated_research_findings,
                    'label': criterion_full_research_data.label
                }
                state.data['research_data'] = research_data
                save_state(state, state_file)

        state.last_completed_stage = Stage.DATA_RESEARCH
        state.data = {**state.data, **dict(research_data=research_data)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.DATA_RESEARCH)

    research_data = state.data['research_data']

    # Analyze Data
    if state.last_completed_stage == Stage.DATA_RESEARCH:
        spinner.start('Analyzing data...')

        items = [research_data[alternative] for alternative in alternatives]
        weights = {c: w for c, w in zip(criteria_names, criteria_weights)}
        scores = topsis_score(items=items,
                              weights=weights,
                              value_mapper=lambda item, criterion: \
                                  normalize_label_value(label=item[criterion]['aggregated']['label'],
                                                        label_list=criteria[criteria_names.index(criterion)]['scale'],
                                                        lower_bound=0.0,
                                                        upper_bound=1.0),
                              best_and_worst_solutions=(
                                  {criterion['name']: {'aggregated': {'label': criterion['scale'][-1]}} for
                                   criterion in criteria},
                                  {criterion['name']: {'aggregated': {'label': criterion['scale'][0]}} for
                                   criterion in criteria}
                              ))

        scored_alternatives = {alternative: score for alternative, score in zip(alternatives, scores)}

        state.last_completed_stage = Stage.DATA_ANALYSIS
        state.data = {**state.data, **dict(scored_alternatives=scored_alternatives)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.DATA_ANALYSIS)

    scored_alternatives = state.data['scored_alternatives']

    # Compile data for presentation
    if state.last_completed_stage == Stage.DATA_ANALYSIS:
        # Aggregate everything into an HTML file for presentation
        enriched_alternatives = []
        for alternative in alternatives:
            alternative_research_data = research_data[alternative]
            alternative_score = scored_alternatives[alternative]

            enriched_alternatives.append({
                'name': alternative,
                'score': alternative_score,
                'criteria_data': alternative_research_data
            })

        spinner.start('Producing report...')

        html = generate_decision_report_as_html(criteria=criteria, alternatives=enriched_alternatives, goal=goal)
        save_html_to_file(html, report_file)

        state.last_completed_stage = Stage.PRESENTATION_COMPILATION

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.PRESENTATION_COMPILATION)

    open_html_file_in_browser(report_file)

    print(state)


if __name__ == '__main__':
    load_dotenv()

Fire(run_decision_assistant)
