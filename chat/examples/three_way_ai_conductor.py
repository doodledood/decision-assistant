from halo import Halo

from chat.backing_stores import InMemoryChatDataBackingStore
from chat.base import Chat
from chat.conductors import LangChainBasedAIChatConductor
from chat.participants import LangChainBasedAIChatParticipant, UserChatParticipant
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

from chat.renderers import TerminalChatRenderer

if __name__ == '__main__':
    load_dotenv()
    chat_model = ChatOpenAI(
        temperature=0.0,
        model='gpt-4-1106-preview'
    )

    spinner = Halo(spinner='dots')
    ai = LangChainBasedAIChatParticipant(name='Assistant',
                                         role='Boring Serious AI Assistant',
                                         chat_model=chat_model,
                                         spinner=spinner)
    rob = LangChainBasedAIChatParticipant(name='Rob', role='Funny Prankster',
                                          personal_mission='Take the lead and try to prank the boring AI. Collaborate '
                                                           'with the user when relevant and make him laugh!',
                                          chat_model=chat_model,
                                          spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai, rob]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants,
        goal='Make the user laugh by pranking the boring AI.',
        speaker_interaction_schema=f'Rob should take the lead and go back and forth with the assistant trying to '
                                   f'prank him big time. Rob can and should talk to the user to get them in on the '
                                   f'prank, however the majority of the prank should be done by Rob. By prank, '
                                   f'I mean the AI should be confused and not know what to do, or laughs at the prank '
                                   f'(funny).',
    )

    chat_conductor = LangChainBasedAIChatConductor(
        chat_model=chat_model,
        spinner=spinner,
        termination_condition='One laugh is enough. Terminate the chat when the user finds a prank funny.'
    )

    chat_conductor.initiate_chat_with_result(chat=chat)
