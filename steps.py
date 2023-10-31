import itertools
import json
import os
from typing import List, Dict, Tuple, Optional, Callable, Generator

import ahpy
import questionary
from halo import Halo
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import RoundRobinChatConductor, LangChainBasedAIChatConductor
from chat.parsing_utils import string_output_to_pydantic
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from chat.renderers import TerminalChatRenderer
from chat.structured_prompt import Section, StructuredPrompt
from presentation import generate_decision_report_as_html, save_html_to_file, open_html_file_in_browser
from chat.web_research import WebSearch
from ranking.ranking import topsis_score, normalize_label_value
from state import DecisionAssistantState


class Criterion(BaseModel):
    name: str = Field(description='The name of the criterion. Example: "Affordability".')
    scale: List[str] = Field(
        description='The scale of the criterion, from worst to best. Labels only. No numerical value, no explainations. Example: "Very Expensive".')


class GoalIdentificationResult(BaseModel):
    goal: str = Field(description='The identified decision-making goal.')


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


def identify_goal(chat_model: ChatOpenAI, state: DecisionAssistantState,
                  tools: Optional[List[BaseTool]] = None, spinner: Optional[Halo] = None):
    if state.data.get('goal') is not None:
        return

    ai = LangChainBasedAIChatParticipant(
        name='Decision-Making Goal Identifier',
        role='Decision-Making Goal Identifier',
        personal_mission='Identify a clear and specific decision-making goal from the user\'s initial vague statement.',
        other_prompt_sections=[
            Section(
                name='Process',
                list=[
                    'Start by greeting the user and asking for their decision-making goal. Example: "Hello, what is your decision-making goal?"',
                    'If the goal is not clear, ask for clarification and refine the goal.',
                    'If the goal is clear, confirm it with the user.',
                ]
            ),
            Section(
                name='User Decision Goal',
                list=[
                    'One and only one decision goal can be identified.',
                    'The goal should be clear and specific.',
                    'The goal should be a decision that can be made by the user.',
                    'No need to go beyond the goal. The next step will be to identify alternatives and criteria for the decision.'
                ]
            ),
            Section(
                name='Last Message',
                list=[
                    'After the goal has been identified, the last message should include the goal.'
                    'It should end with the word TERMINATE at the end of the message to signal the end of the chat.'
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [ai, user]

    chat = Chat(
        goal='Identify a clear and specific decision-making goal.',
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    output = chat_conductor.initiate_chat_with_result(chat=chat)
    goal = string_output_to_pydantic(
        output=output,
        chat_model=chat_model,
        output_schema=GoalIdentificationResult
    )
    goal = goal.goal

    state.data = {**state.data, **dict(goal=goal)}


def identify_alternatives(chat_model: ChatOpenAI, tools: List[BaseTool],
                          state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('alternatives') is not None:
        return

    ai = LangChainBasedAIChatParticipant(
        name='Decision-Making Alternative Consultant',
        role='Decision-Making Alternative Consultant',
        personal_mission='Assist the user in identifying alternatives for the decision-making process.',
        other_prompt_sections=[
            Section(
                name='Interaction Schema',
                list=[
                    'This is the second part of the decision-making process, after the goal has been identified. No need for a greeting.',
                    'Start by asking the user for alternatives they had in mind for the decision.',
                    'Assist the user in generating alternatives if they are unsure or struggle to come up with options or need help researching more ideas. You can use the web search tool and your own knowledge for this.',
                    'List the final list of alternatives and confirm with the user before moving on to the next step.'
                ]
            ),
            Section(
                name='Requirements',
                list=[
                    'At the end of the process there should be at least 2 alternatives and no more than 20.'
                ]
            ),
            Section(
                name='Alternatives',
                list=[
                    'The alternatives should be clear and specific.',
                    'The alternatives should be options that the user can choose from.',
                    'Naming the alternatives should be done in a way that makes it easy to refer to them later on.',
                    'For example, for a goal such as "Decide which school to go to": The alternative "Go to school X" is bad, while "School X" is good.'
                ]
            ),
            Section(
                name='The Last Message',
                list=[
                    'The last response should include the list of confirmed alternatives.',
                    'It should end with the word TERMINATE at the end of the message to signal the end of the chat.'
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai]

    chat = Chat(
        goal='Identify clear alternatives for the decision.',
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    output = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=str(StructuredPrompt(
        sections=[
            Section(name='Goal', text=state.data['goal']),
        ]
    )))
    output = string_output_to_pydantic(
        output=output,
        chat_model=chat_model,
        output_schema=AlternativeListingResult
    )
    alternatives = output.alternatives

    state.data = {**state.data, **dict(alternatives=alternatives)}


def identify_criteria(chat_model: ChatOpenAI, tools: List[BaseTool],
                      state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('criteria') is not None:
        return

    shared_prompt_sections = [
        Section(
            name='Process Stage',
            text='This is the third part of the decision-making process, after the goal and alternatives have been identified. No need for a greeting.'
        ),
    ]

    criteria_brainstormer = LangChainBasedAIChatParticipant(
        name='Decision-Making Criteria Brainstormer',
        role='Decision-Making Criteria Brainstormer',
        personal_mission='Brainstorm and iterate on criteria for the decision-making process for the user.',
        other_prompt_sections=shared_prompt_sections + [
            Section(
                name='Criteria Identification Process',
                list=[
                    'Start by suggesting an initial set of criteria that is as orthogonal, non-overlapping, and comprehensive as possible.',
                    'Iterate on the criteria with the critic until you both are satisfied with them.',
                    'Confirm the criteria with the user and ask for feedback.'
                ]
            ),
            Section(
                name='Requirements',
                list=[
                    'At the end of the process there MUST be at least 1 criterion and no more than 15 criteria.',
                ]
            ),
            Section(
                name='The Last Message',
                list=[
                    'The last response should include the list of confirmed criteria and their respective scales, numbered from 1 to N, where N is the best outcome for the criteria.'
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    criteria_scales_brain_stormer = LangChainBasedAIChatParticipant(
        name='Decision-Making Criteria Scale Brainstormer',
        role='Decision-Making Criteria Scale Brainstormer',
        personal_mission='Brainstorm and iterate on scales for the criteria for the decision-making process for the user.',
        other_prompt_sections=shared_prompt_sections + [
            Section(
                name='Criteria Scale Definition Process',
                list=[
                    'After the criteria have been and the user is satisfied with them, come up with a 2 to 7 point scale for each criterion based on common sense.',
                    'Iterate on the scales until the user is satisfied with them.'
                ],
                sub_sections=[
                    Section(name='Scale Definition', list=[
                        'The scale should be a list of labels only. No numerical values, no explainations. Example: "Very Expensive".',
                        'The scale should be ordered from worst to best. Example: "Very Expensive" should come before "Expensive".',
                        'Make should the values for the scale are roughly evenly spaced out. Example: "Very Expensive" should be roughly as far from "Expensive" as "Expensive" is from "Fair".'
                    ])
                ]
            ),
            Section(
                name='Requirements',
                list=[
                    'Scales MUST be on at least 2-point scale and no more than 7-point scale.'
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    criteria_critic = LangChainBasedAIChatParticipant(
        name='Decision-Making Criteria Critic',
        role='Decision-Making Criteria Critic',
        personal_mission='Critique the criteria and provide feedback on what to improve.',
        other_prompt_sections=shared_prompt_sections + [
            Section(
                name='Criteria Critiquing',
                list=[
                    'When critiquing the criteria, make sure they are orthogonal, non-overlapping, and comprehensive.',
                ],
                sub_sections=[
                    Section(
                        name='Questions to ask yourself',
                        list=[
                            'Are there any criteria that are redundant or duplicated?',
                            'Are there any criteria that are missing to create a comprehensive set of criteria?',
                            'Is the criteria set maximally orthogonal and non-overlapping?',
                            'Are there any criteria that are too subjective or vague?',
                            'Is there at least 1 criterion identified?',
                            'Are there no more than 15 criteria identified?'
                        ]
                    )
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    criteria_scale_critic = LangChainBasedAIChatParticipant(
        name='Decision-Making Criteria Scale Critic',
        role='Decision-Making Criteria Scale Critic',
        personal_mission='Critique the scales for the criteria and provide feedback on what to improve.',
        other_prompt_sections=shared_prompt_sections + [
            Section(
                name='Criteria Scale Critiquing',
                list=[
                    'When critiquing the scales, make sure they are ordered from worst to best, evenly spaced out, and have labels that make sense.',
                ],
                sub_sections=[
                    Section(
                        name='Questions to ask yourself',
                        list=[
                            'Can a scale be simplified such that it is easier to assign a value to a piece of data based on it?',
                            'Is a scale too simple such that it is not useful for the decision-making process?',
                            'Are all the scales on a 2-point to 7-point scale?'
                        ]
                    )
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, criteria_brainstormer, criteria_scales_brain_stormer, criteria_critic, criteria_scale_critic]

    chat = Chat(
        goal='Identify clear well-defined criteria and their respective scales for the decision.',
        speaker_interaction_schema='''1. The Criteria Brainstormer suggests an initial set of criteria based on the user input.
2. The Criteria Critic critiques the criteria and suggests improvements.
3. The Criteria Brainstormer iterates on the criteria until they think they are good enough and ask the user for feedback.
4. If the user is not satisfied with the criteria, go back to step 1, refining the criteria based on the user feedback.
5. If the user is satisfied with the criteria, it\'s time to define the scales for the criteria. Move to step 6.
6. The Criteria Scale Brainstormer suggests an initial set of scales for the criteria.
7. The Criteria Scale Critic critiques the scales and suggests improvements.
8. The Criteria Scale Brainstormer iterates on the scales until they think they are good enough and ask the user for feedback.
9. If the user is not satisfied with the scales, go back to step 6, refining the scales based on the user feedback.
10. If the user is satisfied with the scales, the criteria identification process is complete. The Criteria Brainstormer should present the final list of criteria and their respective scales to the user.
''',
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants
    )

    chat_conductor = LangChainBasedAIChatConductor(
        chat_model=chat_model,
        termination_condition='The criteria and their respective scales have been identified and confirmed by the user.'
    )
    output = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=str(StructuredPrompt(
        sections=[
            Section(name='Goal', text=state.data['goal']),
            Section(name='Alternatives', list=state.data['alternatives']),
        ]
    )))
    output = string_output_to_pydantic(
        output=output,
        chat_model=chat_model,
        output_schema=CriteriaIdentificationResult
    )
    criteria = output.model_dump()['criteria']

    state.data = {**state.data, **dict(criteria=criteria)}


def map_criteria(chat_model: ChatOpenAI, tools: List[BaseTool],
                 state: DecisionAssistantState, spinner: Optional[Halo] = None):
    criteria_mapping = state.data.get('criteria_mapping', {})

    for criterion in state.data['criteria']:
        if criterion['name'] in criteria_mapping:
            continue

        ai = LangChainBasedAIChatParticipant(
            name='Decision-Making Criteria Mapping Consultant',
            role='Decision-Making Criteria Mapping Consultant',
            personal_mission='Develop a concrete, non-ambiguous decision tree for mapping research data onto a given scale for each criterion in a decision-making process.',
            other_prompt_sections=[
                Section(
                    name='Interaction Schema',
                    list=[
                        'This is the fourth part of the decision-making process, after the goal, alternatives, and criteria have been identified. No need for a greeting.',
                        'Start by suggesting a mapping for the criterion at hand and ask the user for feedback.',
                        'Iterate on the mapping until the user is satisfied with it.'
                    ]
                ),
                Section(
                    name='Criterion Mapping',
                    list=[
                        'A mapping is like a decision tree for mapping research data onto a given scale for a criterion.',
                        'Based on the mapping, an autonomous bot should be able to assign a value from the scale to a piece of data.',
                        'Make sure the current mapping does not contradict or interfere with previous mappings or future ones to follow.',
                        'Each mapping should be unique within all the criteria, unless specified by the user.',
                        'A mapping could also be looked at like sub-criteria of the criterion - we don\'t want these duplicated as that makes the results less accurate.'
                    ],
                    sub_sections=[
                        Section(
                            name='Subjective Criterion',
                            list=[
                                'For subjective criteria like "Affordability", engage in a deeper dialogue with the user to understand their preferences and thinking.',
                                'If the criterion is entirely user feeling based, just explain a value mapping by saying something like "User feels very good about this".'
                            ]
                        )
                    ]
                ),
                Section(
                    name='The Last Message',
                    list=[
                        'The last response should include the list of confirmed criteria mapping.',
                        'The list should be formatted like: "1. LABEL_1: EXPLANATION_1\n2. ..." where 1. is the worst option and N. is the best option.',
                        'It should end with the word TERMINATE at the end of the message to signal the end of the chat.'
                    ]
                )
            ],
            tools=tools,
            chat_model=chat_model,
            spinner=spinner)
        user = UserChatParticipant(name='User')
        participants = [user, ai]

        chat = Chat(
            goal='Develop a concrete, non-ambiguous decision tree for mapping research data onto a given scale for each criterion in a decision-making process.',
            backing_store=InMemoryChatDataBackingStore(),
            renderer=TerminalChatRenderer(),
            initial_participants=participants
        )

        chat_conductor = RoundRobinChatConductor()
        output = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=str(StructuredPrompt(
            sections=[
                Section(
                    name='Goal',
                    text=state.data['goal']
                ),
                Section(
                    name='Previously Mapping Criteria',
                    list=[f'{criterion_name}: {criterion_mapping}' for criterion_name, criterion_mapping in
                          criteria_mapping.items()]
                ),
                Section(
                    name='Criteria Left to Map',
                    list=[criterion_name for criterion_name in
                          [criterion['name'] for criterion in state.data['criteria'] if
                           criterion['name'] not in criteria_mapping]]
                ),
                Section(
                    name='Current Criterion',
                    text=criterion['name']
                ),
                Section(
                    name='Current Criterion Scale',
                    list=criterion['scale'],
                    list_item_prefix=None
                )
            ]
        )))
        output = string_output_to_pydantic(
            output=output,
            chat_model=chat_model,
            output_schema=CriterionMappingResult
        )
        criterion_mapping = output.model_dump()['criterion_mapping']

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


def generate_research_questions(chat_model: ChatOpenAI, tools: List[BaseTool],
                                state: DecisionAssistantState, spinner: Optional[Halo] = None):
    if state.data.get('criteria_research_queries') is not None:
        return

    ai = LangChainBasedAIChatParticipant(
        name='Decision-Making Process Researcher',
        role='Decision-Making Process Researcher',
        personal_mission='Generate a template for automated research queries for each criterion, whose answers can be used as context when evaluating alternatives.',
        other_prompt_sections=[
            Section(
                name='Process',
                list=[
                    'This is the fifth part of the decision-making process, after the goal, alternatives, criteria, and criteria mapping have been identified. No need for a greeting.',
                    'For each criterion, generate relevant, orthogonal, and comprehensive set query templates.',
                ],
                list_item_prefix=None
            ),
            Section(
                name='Query Templates',
                list=[
                    'The query templates should capture the essence of the criterion based on the scale and how to assign values.',
                    'The queries should be strategic and aim to minimize the number of questions while maximizing the information gathered.',
                    'The list of queries should include counterfactual queries and make use of all knowledge of information foraging and information literacy.',
                    'Each query template MUST include "{alternative}" in the template to allow for replacement with various alternatives later.',
                    'If a criterion is purely subjective and nothing an be researched on it, it\'s ok to have 0 queries about it.'
                ]
            ),
            Section(
                name='The Last Message',
                list=[
                    'The last response should include the list of research query templates for each criterion.',
                    'It should end with the word TERMINATE at the end of the message to signal the end of the chat.'
                ]
            )
        ],
        tools=tools,
        chat_model=chat_model,
        spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai]

    chat = Chat(
        goal='Generate a template for automated research queries for each criterion, whose answers can be used as context when evaluating alternatives.',
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants,
        max_total_messages=2
    )

    chat_conductor = RoundRobinChatConductor()
    output = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=str(StructuredPrompt(
        sections=[
            Section(name='Goal', text=state.data['goal']),
            Section(name='Alternatives', list=state.data['alternatives']),
            Section(name='Criteria Mappings',
                    sub_sections=[
                        Section(name=criterion_name, text=criterion_mapping) for criterion_name, criterion_mapping in
                        state.data['criteria_mapping'].items()
                    ]),
        ]
    )))
    output = string_output_to_pydantic(
        output=output,
        chat_model=chat_model,
        output_schema=CriteriaResearchQueriesResult
    )
    criteria_research_queries = output.model_dump()['criteria_research_queries']

    state.data = {**state.data, **dict(criteria_research_queries=criteria_research_queries)}


def perform_research(chat_model: ChatOpenAI, web_search: WebSearch, n_search_results: int,
                     tools: List[BaseTool], state: DecisionAssistantState,
                     spinner: Optional[Halo] = None,
                     fully_autonomous: bool = True):
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

            ai = LangChainBasedAIChatParticipant(
                name='Decision-Making Process Researcher',
                role='Decision-Making Process Researcher',
                personal_mission='Refine research findings through user interaction and assign an accurate label based on data, user input, and criteria mapping.',
                other_prompt_sections=[
                    Section(
                        name='Process',
                        list=[
                            'This is the sixth part of the decision-making process, after the goal, alternatives, criteria, criteria mapping, and research queries have been identified. No need for a greeting.',
                            'Present the researched data to the user and assign a preliminary label & ask for feedback',
                            'Revise the research findings based on user input, until the user is satisfied with the findings and label.',
                        ],
                    ),
                    Section(
                        name='Research Presentation',
                        list=[
                            'Maintain original findings if no new user input.',
                            'Mention the sources of the research findings as inline links.'
                        ]
                    ),
                    Section(
                        name='Label Assignment',
                        list=[
                            'Assign one label per criterion per alternative based on scale and value assignment rules. A label should be a string only, e.g., "Very Expensive".',
                            'If unclear, make an educated guess based on data and user input.'
                        ]
                    ),
                    Section(
                        name='The First Message',
                        list=[
                            'Your first message should look something like this: "Here is what I found about {alternative} for {criterion}:\n\n{research_findings}\n\nBecause {reason_for_label_assignment}, I think the label for {alternative} for {criterion} should be {label}. What do you think? Do you have anything else to add, clarify or change that might affect this label?"'
                        ]
                    ),
                    Section(
                        name='The Last Message',
                        list=[
                            'The last response should include the refined research findings for a criterion\'s alternative in rich markdown format with all the citations and links inline.',
                            'Does not include conversational fluff. Think about it like a research report.',
                            'It should end with the word TERMINATE at the end of the message to signal the end of the chat.'
                        ]
                    )
                ],
                tools=tools,
                chat_model=chat_model,
                spinner=spinner)
            user = UserChatParticipant(name='User')
            participants = [user, ai]

            chat = Chat(
                goal='Generate a template for automated research queries for each criterion, whose answers can be used as context when evaluating alternatives.',
                backing_store=InMemoryChatDataBackingStore(),
                renderer=TerminalChatRenderer(),
                initial_participants=participants,
                max_total_messages=2 if fully_autonomous else None
            )

            chat_conductor = RoundRobinChatConductor()
            output = chat_conductor.initiate_chat_with_result(chat=chat, initial_message=str(StructuredPrompt(
                sections=[
                    Section(name='Goal', text=state.data['goal']),
                    Section(name='Alternatives', list=state.data['alternatives']),
                    Section(name='Criteria Mappings',
                            sub_sections=[
                                Section(name=criterion_name, text=criterion_mapping) for
                                criterion_name, criterion_mapping in
                                state.data['criteria_mapping'].items()
                            ]),
                    Section(name='Research Findings',
                            sub_sections=[
                                Section(name=query, text=answer) for query, answer in
                                alternative_criterion_research_data[
                                    'raw'].items()
                            ])
                ]
            )))
            criterion_full_research_data = string_output_to_pydantic(
                output=output,
                chat_model=chat_model,
                output_schema=AlternativeCriteriaResearchFindingsResult
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
