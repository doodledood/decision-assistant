from halo import Halo

from chat.base import Chat
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    chat_model = ChatOpenAI(
        temperature=0.0,
        model='gpt-4-0613'
    )

    spinner = Halo(spinner='dots')
    ai = LangChainBasedAIChatParticipant(name='Assistant',
                                         role='Boring Serious AI Assistant',
                                         chat_model=chat_model,
                                         spinner=spinner)
    rob = LangChainBasedAIChatParticipant(name='Rob', role='Funny Prankster',
                                          mission='Collaborate with the user to prank the boring AI. Yawn.',
                                          chat_model=chat_model,
                                          spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai, rob]

    main_chat = Chat(
        initial_participants=participants
    )

    chat_conductor = LangChainBasedAIChatParticipant(
        chat_model=chat_model,
        speaker_interaction_schema=f'Rob should take the lead and go back and forth with the assistant trying to prank him big time. Rob can and should talk to the user to get them in on the prank, however the majority of the prank should be done by Rob. By prank, I mean the AI should be confused and not know what to do, or laughs at the prank (funny).',
        termination_condition=f'Terminate the chat when the is successfuly pranked, or is unable to be pranked or does not go along with the pranks within a 2 tries, OR if the user asks you to terminate the chat.',
        spinner=spinner
    ),
    main_chat.initiate_chat_with_result()
