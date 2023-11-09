from chat.base import ChatRenderer, Chat, ChatMessage


class TerminalChatRenderer(ChatRenderer):
    def render_new_chat_message(self, chat: Chat, message: ChatMessage):
        if chat.hide_messages:
            return

        sender = chat.get_active_participant_by_name(message.sender_name)
        if sender is None:
            symbol = '❓'

            print(f'{symbol} {message.sender_name}: {message.content}')
        else:
            if sender.messages_hidden:
                return

            if chat.name is None:
                print(f'{str(sender)}: {message.content}')
            else:
                print(f'{chat.name} > {str(sender)}: {message.content}')
