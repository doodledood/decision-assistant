from typing import List, Optional, Callable, Type, TypeVar, Dict, Any

from halo import Halo
from pydantic import BaseModel, Field

TOutputSchema = TypeVar("TOutputSchema", bound=BaseModel)



terminate_now_message_content = 'Please now "_terminate" immediately with the result of your mission.'


if __name__ == '__main__':
    from langchain.chat_models import ChatOpenAI
    from dotenv import load_dotenv

    load_dotenv()
    chat_model = ChatOpenAI(
        temperature=0.0,
        model='gpt-4-0613'
    )

    spinner = Halo(spinner='dots')

    # ai = AIChatParticipant(name='Assistant',
    #                        role='Boring Serious AI Assistant',
    #                        chat_model=chat_model,
    #                        spinner=spinner)
    # rob = AIChatParticipant(name='Rob', role='Funny Prankster',
    #                         mission='Collaborate with the user to prank the boring AI. Yawn.',
    #                         chat_model=chat_model,
    #                         spinner=spinner)
    # user = UserChatParticipant(name='User')
    # participants = [user, ai, rob]
    #
    # main_chat = Chat(
    #     initial_participants=participants,
    #     chat_conductor=AIChatConductor(
    #         chat_model=chat_model,
    #         speaker_interaction_schema=f'Rob should take the lead and go back and forth with the assistant trying to prank him big time. Rob can and should talk to the user to get them in on the prank, however the majority of the prank should be done by Rob. By prank, I mean the AI should be confused and not know what to do, or laughs at the prank (funny).',
    #         termination_condition=f'Terminate the chat when the is successfuly pranked, or is unable to be pranked or does not go along with the pranks within a 2 tries, OR if the user asks you to terminate the chat.',
    #         spinner=spinner
    #     ),
    # )
    # main_chat.initiate_chat_with_result()

    # ai = AIChatParticipant(name='AI', role='Math Expert',
    #                        mission='Solve the user\'s math problem (only one). Respond with the correct answer and end with the word "TERMINATE"',
    #                        chat_model=chat_model, spinner=spinner)
    # user = UserChatParticipant(name='User')
    # participants = [user, ai]
    #
    #
    # class MathResult(BaseModel):
    #     result: float = Field(description='The result of the math problem.')
    #
    #
    # main_chat = Chat(initial_participants=participants)
    # parsed_output = string_output_to_pydantic(
    #     output=main_chat.initiate_chat_with_result(),
    #     chat_model=chat_model,
    #     output_schema=MathResult,
    #     spinner=spinner
    # )
    #
    # print(f'Result: {dict(parsed_output)}')

    #     criteria_generation_team = GroupBasedChatParticipant(
    #         chat=Chat(
    #             initial_participants=[
    #                 LangChainBasedAIChatParticipant(
    #                     name='Tom',
    #                     role='Criteria Generation Team Leader',
    #                     mission=f'Delegate to your team and respond back with comprehensive, orthogonal, well-researched criteria for a decision-making problem.',
    #                     other_prompt_sections={
    #                         'Last Message': '''- Once the criteria set is finalized you will send the last message.
    # - This last message will be sent to the external conversation verbatim. Act as if you are responding directly to the other chat yourself.
    # - Ignore the group and their efforts in the last message as this isn't relevant for the other chat.
    # '''
    #                     },
    #                     chat_model=chat_model,
    #                     spinner=spinner),
    #                 LangChainBasedAIChatParticipant(
    #                     name='Rob',
    #                     role='Criteria Generator',
    #                     mission='Think from first principles about the decision-making problem, and come up with orthogonal, compresive list of criteria. Iterate on it, as needed.',
    #                     other_prompt_sections={
    #                         'Receiving Feedback': 'John might criticize your criteria and provide counterfactual evidence to support his criticism. You should respond to his criticism and provide counter-counterfactual evidence to support your response, if applicable.'
    #                     },
    #                     chat_model=chat_model,
    #                     spinner=spinner),
    #                 LangChainBasedAIChatParticipant(
    #                     name='John',
    #                     role='Criteria Generation Critic',
    #                     mission='Think from frist principles and collaborate with Rob to come up with a comprehensive, orthogonal list of criteria. Criticize Rob\'s criteria and provide counterfactual evidence to support your criticism. Are some criteria overlapping and need to be merged? Is some criterion too general and need to be broken down? Are there criteria missing? Is the naming of each criteria suitable and reflects that a higher value is better? Iterate on it, as needed.',
    #                     other_prompt_sections={
    #                         'Receiving Feedback': 'Rob might criticize your criticism and provide counter-counterfactual evidence to support his response, if applicable.'
    #                     },
    #                     chat_model=chat_model,
    #                     spinner=spinner),
    #             ],
    #             chat_conductor=LangChainBasedAIChatConductor(
    #                 chat_model=chat_model,
    #                 speaker_interaction_schema='The team leader initiates the conversation about the criteria. Rob and John will go back and forth, refining and improving the criteria set until they both think the set cannot be improved anymore. Then, finally, once they both agree the set is good enough, the team leader responds with a message to the external conversation with the final criteria set.',
    #                 termination_condition='Terminate the chat when the team leader thinks the criteria set is good enough, or if the team leader asks you to terminate the chat.',
    #                 spinner=spinner
    #             ),
    #             description='This chat room is a group chat for the criteria generation team. Everybody can talk to everybody else. The goal is to generate a list of criteria for a decision-making problem.',
    #         ),
    #         spinner=spinner
    #     )
    #     user = UserChatParticipant(name='User')
    #     participants = [user, criteria_generation_team]
    #
    #     main_chat = Chat(initial_participants=participants)
    #     result = main_chat.initiate_chat_with_result(
    #         initial_message="Please generate a list of criteria for choosing the breed of my next puppy.",
    #     ),
    #
    #     print(f'Result: {result}')

    # from datetime import datetime
    # from pydantic import Field
    #
    #
    # def get_time():
    #     return datetime.utcnow().isoformat()
    #
    #
    # class GetTime(BaseModel):
    #     """Returns the current time in UTC."""
    #
    #     result: str = Field(description='The current time in UTC.')
    #
    #
    # now = get_time()
    # ai = LangChainBasedAIChatParticipant(name='Assistant',
    #                                      mission=f'Be a helpful, time-showing, AI assistant. Always tell them the UTC time (in a user friendly format) when you respond and how much time you have left to work. You stop working after 1 minute has passed from when you started working at {now}. When the time comes, let the user know you will stop working and say goodbye.',
    #                                      chat_model=chat_model,
    #                                      functions={
    #                                          GetTime.__name__: lambda _: get_time()
    #                                      },
    #                                      chat_model_args={
    #                                          'functions': [pydantic_to_openai_function(GetTime)]
    #                                      },
    #                                      spinner=spinner)
    # user = UserChatParticipant(name='User')
    # participants = [user, ai]
    #
    # main_chat = Chat(
    #     initial_participants=participants,
    #     chat_conductor=LangChainBasedAIChatConductor(
    #         chat_model=chat_model,
    #         termination_condition=f'Chat started at {now}. Terminate the chat when the assistant stops working (He will let it be known in the chat), OR if the user decides to terminate the chat by ending their sentence with "TERMINATE".',
    #         spinner=spinner
    #     ),
    # )
    # main_chat.initiate_chat_with_result()

    # ai = LangChainBasedAIChatParticipant(name='Assistant',
    #                                      role='Boring Serious AI Assistant',
    #                                      chat_model=chat_model,
    #                                      spinner=spinner)
    user = UserChatParticipant(name='User')
    # participants = [user, ai]

    chat_conductor = LangChainBasedAIChatConductor(
        chat_model=chat_model,
        spinner=spinner
    )
    chat = Chat(
        goal='Come up with a plan for the user to invest their money. The goal is to maximize wealth over the long-term, while minimizing risk (Equivalent to maximizing the geometric return of wealth).',
        initial_participants=[user],
        chat_composition_generator= \
            LangChainBasedAIChatCompositionGenerator(
                chat_model=chat_model,
                spinner=spinner
            )
    )
    result = chat_conductor.initiate_chat_with_result(chat=chat)

    print(f'Result: {result}')
