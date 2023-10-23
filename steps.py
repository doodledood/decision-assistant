import itertools
import json
import os
from typing import List, Dict, Tuple, Optional, Callable, Generator

import ahpy
import questionary
from halo import Halo
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import Tool
from pydantic.v1 import BaseModel, Field

import system_prompts
from chat import chat
from presentation import generate_decision_report_as_html, save_html_to_file, open_html_file_in_browser
from research import WebSearch
from research.ranking import topsis_score, normalize_label_value
from state import DecisionAssistantState


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


class CriteriaResearchQueriesResult(BaseModel):
    criteria_research_queries: Dict[str, List[str]] = Field(
        description='The research queries for each criteria. Key is the criterion name, value is a list of research queries for that criterion.')


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
                                       previous_comparisons: Optional[List[Tuple[Tuple[str, str], float]]] = None,
                                       on_question_asked: Optional[Callable[[Tuple[str, str], float], None]] = None) \
        -> Generator[Tuple[Tuple[str, str], float], None, None]:
    choices = {
        'Absolutely less important': 1 / 9,
        'A lot less important': 1 / 7,
        'Notably less important': 1 / 5,
        'Slightly less important': 1 / 3,
        'Just as important': 1,
        'Slightly more important': 3,
        'Notably more important': 5,
        'A lot more important': 7,
        'Absolutely more important': 9
    }
    ordered_choice_names = [choice[0] for choice in sorted(choices.items(), key=lambda x: x[1])]

    comparisons = dict(previous_comparisons)
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

        yield labels, value


def identify_goal(chat_model: ChatOpenAI, default_tools_with_web_search: List[Tool],
                  state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('goal') is not None:
        return

    goal = chat(
        chat_model=chat_model,
        messages=[
            SystemMessage(content=system_prompts.goal_identification_system_prompt),
            HumanMessage(content="Hey")
        ],
        tools=default_tools_with_web_search,
        spinner=spinner
    )

    state.data = {**state.data, **dict(goal=goal)}


def identify_alternatives(chat_model: ChatOpenAI, default_tools_with_web_search: List[Tool],
                          state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('alternatives') is not None:
        return

    alternatives = chat(
        chat_model=chat_model,
        messages=[
            SystemMessage(content=system_prompts.alternative_listing_system_prompt),
            HumanMessage(content=f'# GOAL\n{state.data["goal"]}'),
        ],
        tools=default_tools_with_web_search,
        result_schema=AlternativeListingResult,
        spinner=spinner
    )
    alternatives = alternatives.dict()['alternatives']

    state.data = {**state.data, **dict(alternatives=alternatives)}


def identify_criteria(chat_model: ChatOpenAI, default_tools_with_web_search: List[Tool],
                      state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('criteria') is not None:
        return

    criteria = chat(
        chat_model=chat_model,
        messages=[
            SystemMessage(content=system_prompts.criteria_identification_system_prompt),
            HumanMessage(content=f'# GOAL\n{state.data["goal"]}'),
        ],
        tools=default_tools_with_web_search,
        result_schema=CriteriaIdentificationResult,
        spinner=spinner
    )
    criteria = criteria.dict()['criteria']

    state.data = {**state.data, **dict(criteria=criteria)}


def map_criteria(chat_model: ChatOpenAI, default_tools_with_web_search: List[Tool],
                 state: DecisionAssistantState, spinner: Optional[Halo] = None):
    criteria_mapping = state.data.get('criteria_mapping', {})

    for criterion in state.data['criteria']:
        if criterion['name'] in criteria_mapping:
            continue

        scale_str = '\n'.join([f'{i + 1}. {scale_value}' for i, scale_value in enumerate(criterion['scale'])])
        criterion_mapping = chat(
            chat_model=chat_model,
            messages=[
                SystemMessage(content=system_prompts.criterion_mapping_system_prompt),
                HumanMessage(
                    content=f'# GOAL\n{state.data["goal"]}\n\n# CRITERION NAME\n{criterion["name"]}\n\n# CRITERION SCALE\n{scale_str}'),
            ],
            tools=default_tools_with_web_search,
            result_schema=CriterionMappingResult,
            spinner=spinner
        )
        criterion_mapping = criterion_mapping.dict()['criterion_mapping']
        criteria_mapping[criterion['name']] = criterion_mapping

        state.data = {**state.data,
                      **dict(criteria_mapping=criteria_mapping)}
        yield state

    state.data = {**state.data, **dict(criteria_mapping=criteria_mapping)}


def prioritize_criteria(state: DecisionAssistantState):
    if state.data.get('criteria_weights') is not None:
        return

    criteria_comparisons = state.data.get('criteria_comparisons', {})
    criteria_comparisons = {tuple(json.loads(labels)): value for labels, value in criteria_comparisons.items()}
    criteria_comparisons = list(criteria_comparisons.items())

    criteria_names = [criterion['name'] for criterion in state.data['criteria']]

    for labels, value in gather_unique_pairwise_comparisons(criteria_names,
                                                            previous_comparisons=criteria_comparisons):
        criteria_comparisons.append((labels, value))

        state.data = {**state.data, **dict(
            criteria_comparisons={json.dumps(labels): value for labels, value in criteria_comparisons})}
        yield state

    state.data['criteria_weights'] = ahpy.Compare('Criteria', dict(criteria_comparisons)).target_weights


def generate_research_questions(chat_model: ChatOpenAI, default_tools_with_web_search: List[Tool],
                                state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('criteria_research_queries') is not None:
        return

    criteria_mapping_str = '\n\n'.join(
        [f'## {criterion_name}\n{criterion_mapping}' for i, (criterion_name, criterion_mapping) in
         enumerate(state.data['criteria_mapping'].items())])
    alternatives_str = '\n\n'.join([f'## {alternative}' for i, alternative in enumerate(state.data['alternatives'])])

    criteria_research_queries = chat(
        chat_model=chat_model,
        messages=[
            SystemMessage(content=system_prompts.criteria_research_questions_system_prompt),
            HumanMessage(
                content=f'# GOAL\n{state.data["goal"]}\n\n# CRITERIA MAPPING\n{criteria_mapping_str}# ALTERNATIVES\n{alternatives_str}'),
        ],
        tools=default_tools_with_web_search,
        result_schema=CriteriaResearchQueriesResult,
        get_immediate_answer=True,
        spinner=spinner
    )
    criteria_research_queries = criteria_research_queries.dict()['criteria_research_queries']

    state.data = {**state.data, **dict(criteria_research_queries=criteria_research_queries)}


def perform_research(chat_model: ChatOpenAI, web_search: WebSearch, n_search_results: int,
                     default_tools_with_web_search: List[Tool], state: DecisionAssistantState,
                     spinner: Optional[Halo] = None,
                     fully_autonomous: bool = False):
    research_data = state.data.get('research_data')
    if research_data is None:
        research_data = {}

    for alternative in state.data['alternatives']:
        alternative_research_data = research_data.get(alternative)

        if alternative_research_data is None:
            alternative_research_data = {}

        for i, criterion in enumerate(state.data['criteria']):
            criterion_name = criterion['name']
            criterion_research_questions = state.data['criteria_research_queries'][criterion_name]
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

                    if spinner:
                        spinner.warn(f'No answer found for query "{query}".')
                else:
                    alternative_criterion_research_data['raw'][query] = answer

                alternative_research_data[criterion_name] = alternative_criterion_research_data
                research_data[alternative] = alternative_research_data
                state.data['research_data'] = research_data

                yield state

    # Do this separately, so all the automated research runs entirely before the user is asked to discuss the findings
    for alternative in state.data['alternatives']:
        alternative_research_data = research_data.get(alternative)

        if alternative_research_data is None:
            alternative_research_data = {}

        for i, criterion in enumerate(state.data['criteria']):
            criterion_name = criterion['name']

            # Present research data, discuss, aggregate, assign a proper label, and confirm with the user
            criterion_mapping = state.data['criteria_mapping'][criterion_name]
            alternative_criterion_research_data = alternative_research_data[criterion_name]

            # Already researched and aggregated, skip
            if alternative_criterion_research_data['aggregated'] != {}:
                continue

            alternative_criterion_research_data_str = '\n\n'.join(
                [f'## {query}\n{answer}' for query, answer in alternative_criterion_research_data['raw'].items()]
            )

            criterion_full_research_data = chat(
                chat_model=chat_model,
                messages=[
                    SystemMessage(
                        content=system_prompts.alternative_criteria_research_system_prompt
                    ),
                    HumanMessage(
                        content=f'# GOAL\n{state.data["goal"]}\n\n# ALTERNATIVE\n{alternative}\n\n# CRITERION MAPPING\n{criterion_mapping}\n\n# RESEARCH FINDINGS\n{alternative_criterion_research_data_str}'),
                ],
                tools=default_tools_with_web_search,
                result_schema=AlternativeCriteriaResearchFindingsResult,
                spinner=spinner,
                get_immediate_answer=fully_autonomous
            )

            research_data[alternative][criterion_name]['aggregated'] = {
                'findings': criterion_full_research_data.updated_research_findings,
                'label': criterion_full_research_data.label
            }
            state.data['research_data'] = research_data

            yield state

    state.data = {**state.data, **dict(research_data=research_data)}


def analyze_data(state: DecisionAssistantState):
    if state.data.get('scored_alternatives') is not None:
        return

    items = [state.data['research_data'][alternative] for alternative in state.data['alternatives']]

    criteria_weights = state.data['criteria_weights']
    criteria_names = [criterion['name'] for criterion in state.data['criteria']]

    scores = topsis_score(items=items,
                          weights=criteria_weights,
                          value_mapper=lambda item, criterion: \
                              normalize_label_value(label=item[criterion]['aggregated']['label'],
                                                    label_list=state.data['criteria'][
                                                        criteria_names.index(criterion)]['scale'],
                                                    lower_bound=0.0,
                                                    upper_bound=1.0),
                          best_and_worst_solutions=(
                              {criterion['name']: {'aggregated': {'label': criterion['scale'][-1]}} for
                               criterion in state.data['criteria']},
                              {criterion['name']: {'aggregated': {'label': criterion['scale'][0]}} for
                               criterion in state.data['criteria']}
                          ))
    scored_alternatives = {alternative: score for alternative, score in zip(state.data['alternatives'], scores)}

    state.data = {**state.data, **dict(scored_alternatives=scored_alternatives)}


def compile_data_for_presentation(state: DecisionAssistantState, report_file: str):
    if os.path.exists(report_file):
        return

    enriched_alternatives = []
    for alternative in state.data['alternatives']:
        alternative_research_data = state.data['research_data'][alternative]
        alternative_score = state.data['scored_alternatives'][alternative]

        enriched_alternatives.append({
            'name': alternative,
            'score': alternative_score,
            'criteria_data': alternative_research_data
        })

    html = generate_decision_report_as_html(
        criteria=state.data['criteria'],
        criteria_weights=state.data['criteria_weights'],
        alternatives=enriched_alternatives,
        goal=state.data['goal'])
    save_html_to_file(html, report_file)


def present_report(state: DecisionAssistantState, report_file: str):
    open_html_file_in_browser(report_file)
