import enum
import json
from typing import Optional, List, Tuple, Any, Dict

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
from research import create_web_search_tool, WebSearch
from research.page_analyzer import OpenAIChatPageQueryAnalyzer
from research.page_retriever import ScraperAPIPageRetriever
from research.ranking import topsis_score, normalize_label_value
from research.search import GoogleSerperSearchResultsProvider


class Criterion(BaseModel):
    name: str = Field(description='The name of the criterion. Example: "Affordability".')
    scale: List[str] = Field(
        description='The scale of the criterion, from worst to best. Labels only. No numerical value, no explainations. Example: "Very Expensive".')


class CriteriaIdentificationResult(BaseModel):
    criteria: List[Criterion] = Field(description='The identified criteria for evaluating the decision.')


class AlternativeListingResult(BaseModel):
    alternatives: List[str] = Field(description='The identified alternatives for the decision.')


class CriteriaMappingResult(BaseModel):
    criteria_mapping: List[str] = Field(
        description='An explaination for each criterion on how to assign a value from the scale to a piece of data. Ordered in the same way as the criteria.')


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


class Stage(enum.IntEnum):
    GOAL_IDENTIFICATION = 0
    ALTERNATIVE_LISTING = 1
    CRITERIA_IDENTIFICATION = 2
    CRITERIA_MAPPING = 3
    CRITERIA_PRIORITIZATION = 4
    CRITERIA_RESEARCH_QUESTIONS_GENERATION = 5
    DATA_RESEARCH = 6
    DATA_ANALYSIS = 7
    PRESENTATION_COMPILATION = 8


class DecisionAssistantState(BaseModel):
    last_completed_stage: Optional[Stage] = Field(description='The current stage of the decision-making process.')
    data: Any = Field(description='The data collected so far.')


class Alternative(BaseModel):
    name: str = Field(description='The name of the alternative.')
    criteria_data: Optional[Dict[str, Tuple[str, int]]] = Field(
        description='The research data collected for each criterion for this alternative. Key is the name of the criterion. Value is a tuple of the research data as text and the assigned value based on the scale of the criterion.')


def save_state(state: DecisionAssistantState, state_file: Optional[str]):
    if state_file is None:
        return

    data = state.dict()
    with open(state_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_state(state_file: Optional[str]) -> Optional[DecisionAssistantState]:
    if state_file is None:
        return None

    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
            return DecisionAssistantState.parse_obj(data)
    except FileNotFoundError:
        return None


def mark_stage_as_done(stage: Stage, halo: Optional[Halo] = None):
    if halo is None:
        halo = Halo(spinner='dots')

    stage_text = {
        Stage.GOAL_IDENTIFICATION: 'Goal identified.',
        Stage.ALTERNATIVE_LISTING: 'Alternatives listed.',
        Stage.CRITERIA_IDENTIFICATION: 'Criteria identified.',
        Stage.CRITERIA_MAPPING: 'Criteria mapped.',
        Stage.CRITERIA_PRIORITIZATION: 'Criteria prioritized.',
        Stage.CRITERIA_RESEARCH_QUESTIONS_GENERATION: 'Research questions generated.',
        Stage.DATA_RESEARCH: 'Data researched.',
        Stage.DATA_ANALYSIS: 'Data analyzed.',
        Stage.PRESENTATION_COMPILATION: 'Presentation compiled.'
    }[stage]
    halo.succeed(stage_text)


def save_and_mark_stage_as_done(state: DecisionAssistantState, state_file: Optional[str]):
    with Halo(text='Saving state...', spinner='dots') as spinner:
        save_state(state, state_file)

        mark_stage_as_done(state.last_completed_stage, spinner)


def run_decision_assistant(goal: Optional[str] = None, llm_temperature: float = 0.0, llm_model: str = 'gpt-4-0613',
                           fast_llm_model: str = 'gpt-3.5-turbo-16k-0613',
                           state_file: Optional[str] = 'state.json', streaming: bool = False,
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

    # Map criteria
    if state.last_completed_stage == Stage.CRITERIA_IDENTIFICATION:
        spinner.start('Mapping criteria...')

        criteria_mapping = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_mapping_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}\n\n# CRITERIA\n{criteria}'),
        ], tools=default_tools_with_web_search, result_schema=CriteriaMappingResult, spinner=spinner)
        criteria_mapping = criteria_mapping.dict()['criteria_mapping']

        state.last_completed_stage = Stage.CRITERIA_MAPPING
        state.data = {**state.data, **dict(criteria_mapping=criteria_mapping)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.CRITERIA_MAPPING)

    criteria_mapping = state.data['criteria_mapping']

    # Prioritize criteria
    if state.last_completed_stage == Stage.CRITERIA_MAPPING:
        spinner.start('Prioritizing criteria...')

        criteria_weights = chat(chat_model=chat_model, messages=[
            SystemMessage(content=system_prompts.criteria_prioritization_system_prompt),
            HumanMessage(content=f'# GOAL\n{goal}\n\n# CRITERIA\n{criteria}\n\n# CRITERIA MAPPING\n{criteria_mapping}'),
        ], tools=default_tools_with_web_search, result_schema=CriteriaPrioritizationResult, spinner=spinner)
        criteria_weights = criteria_weights.dict()['criteria_weights']

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
                criterion_mapping = criteria_mapping[i]
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
        criteria_names = [criterion['name'] for criterion in criteria]
        weights = {c: w for c, w in zip(criteria_names, criteria_weights)}
        scores = topsis_score(items=items, weights=weights, value_mapper=lambda item, criterion: \
            normalize_label_value(label=item[criterion]['aggregated']['label'],
                                  label_list=criteria[criteria_names.index(criterion)]['scale'],
                                  lower_bound=1.0,
                                  upper_bound=100.0))

        scored_alternatives = {alternative: score for alternative, score in zip(alternatives, scores)}

        state.last_completed_stage = Stage.DATA_ANALYSIS
        state.data = {**state.data, **dict(scored_alternatives=scored_alternatives)}

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.DATA_ANALYSIS)

    scored_alternatives = state.data['scored_alternatives']

    # Compile data for presentation
    if state.last_completed_stage == Stage.DATA_ANALYSIS:
        # Combine all alternative bits into an array of alternative objects
        enriched_alternatives = []
        for alternative in alternatives:
            alternative_research_data = research_data[alternative]
            alternative_score = scored_alternatives[alternative]

            enriched_alternatives.append({
                'name': alternative,
                'score': alternative_score,
                'criteria_data': alternative_research_data
            })

        # Aggregate everything into a markdown file for presentation
        criteria_names = [criterion['name'] for criterion in criteria]

        def get_criteria_value_string_for_alternative(alternative):
            criterion_values = []
            for criterion in criteria_names:
                criterion_value = alternative['criteria_data'][criterion]['aggregated']['label']
                criterion_values.append(criterion_value)

            return ' | '.join(criterion_values)

        alternative_lines = [
            f'| {alternative["name"]} | {get_criteria_value_string_for_alternative(alternative)} | {int(round(alternative["score"], 2) * 100)}% |'
            for alternative in sorted(enriched_alternatives, key=lambda a: a['score'], reverse=True)
        ]
        alternative_lines_str = '\n'.join(alternative_lines)

        def scale_to_str(scale):
            return '\n'.join([f'{i + 1}. {label}' for i, label in enumerate(scale)])

        criteria_definition_str = '\n'.join(
            [f'## {criterion["name"]} (Weight = {criterion_weight})\n\n{scale_to_str(criterion["scale"])}\n\n' for
             criterion, criterion_weight in zip(criteria, criteria_weights)]
        )

        def criterion_data_to_full_findings_str(criterion_name, criterion_data):
            return f'''### {criterion_name}

{criterion_data['aggregated']['findings']}

Assigned label: **{criterion_data['aggregated']['label']}**'''

        def get_alternative_criteria_findings_str(alternative):
            return '\n\n'.join(
                [criterion_data_to_full_findings_str(criterion_name, alternative['criteria_data'][criterion_name]) for
                 criterion_name in criteria_names])

        full_research_findings_str = '\n\n'.join(
            [
                f'''## {alternative["name"]}

{get_alternative_criteria_findings_str(alternative)}'''
                for alternative in enriched_alternatives]
        )

        markdown = f'''# GOAL
{goal}

# ALTERNATIVES
| Alternative | {" | ".join(criteria_names)} | Score |
| --- | {" | ".join(["---" for _ in range(len(criteria_names))])} | --- |
{alternative_lines_str}

# CRITERIA
{criteria_definition_str}

# FULL RESEARCH FINDINGS
{full_research_findings_str}'''

        def convert_markdown_to_html(md):
            decorated_html = chat(chat_model=fast_chat_model, messages=[
                SystemMessage(content=system_prompts.convert_markdown_to_html_system_prompt),
                HumanMessage(content=md),
            ], return_first_response=True, spinner=spinner)

            return decorated_html

        spinner.start('Converting generated markdown into readable HTML...')

        partial_html = convert_markdown_to_html(markdown)

        render_partial_html_to_file(partial_html=partial_html, title='Decision Report.html',
                                    filename='decision_report.html')

        state.last_completed_stage = Stage.PRESENTATION_COMPILATION

        save_and_mark_stage_as_done(state, state_file)
    else:
        mark_stage_as_done(Stage.PRESENTATION_COMPILATION)

    print(state)


def render_partial_html_to_file(partial_html: str, title: str, filename: str) -> None:
    # Full HTML with Semantic UI styling
    full_html = f'''<!DOCTYPE html>
<html>
<head>
  <title>{title}</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.5.0/semantic.min.css" integrity="sha512-KXol4x3sVoO+8ZsWPFI/r5KBVB/ssCGB5tsv2nVOKwLg33wTFP3fmnXa47FdSVIshVTgsYk/1734xSk9aFIa4A==" crossorigin="anonymous" referrerpolicy="no-referrer" />
  <script src="https://code.jquery.com/jquery-3.1.1.min.js"
          integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8="
          crossorigin="anonymous"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.5.0/semantic.min.js" integrity="sha512-Xo0Jh8MsOn72LGV8kU5LsclG7SUzJsWGhXbWcYs2MAmChkQzwiW/yTQwdJ8w6UA9C6EVG18GHb/TrYpYCjyAQw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <style>
    body {{
      font-size: 18px;
    }}
  </style>
</head>
<body>
  <div class="ui container" style="margin-top: 20px;">
    <div class="ui raised very padded text segment">
      <div class="ui header">{title}</div>
      {partial_html}
    </div>
  </div>
</body>
</html>'''

    # Write the full HTML to a file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_html)


if __name__ == '__main__':
    load_dotenv()

Fire(run_decision_assistant)
