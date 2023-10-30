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
        model='gpt-4-0613'
    )

    spinner = Halo(spinner='dots')
    ai = LangChainBasedAIChatParticipant(name='Assistant',
                                         role='Boring Serious AI Assistant',
                                         chat_model=chat_model,
                                         spinner=spinner)
    rob = LangChainBasedAIChatParticipant(name='Rob', role='Funny Prankster',
                                          personal_mission='Collaborate with the user to prank the boring AI. Yawn. Make the user laugh!',
                                          chat_model=chat_model,
                                          spinner=spinner)
    user = UserChatParticipant(name='User')
    participants = [user, ai, rob]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants,
        goal='Prank the AI. Have a fun and funny chat for everyone.',
        speaker_interaction_schema=f'Rob should take the lead and go back and forth with the assistant trying to prank him big time. Rob can and should talk to the user to get them in on the prank, however the majority of the prank should be done by Rob. By prank, I mean the AI should be confused and not know what to do, or laughs at the prank (funny).',
    )

    chat_conductor = LangChainBasedAIChatConductor(
        chat_model=chat_model,
        spinner=spinner
    )

    chat_conductor.initiate_chat_with_result(chat=chat)
