from chat.base import ChatRenderer, Chat, ChatMessage


class TerminalChatRenderer(ChatRenderer):
    def render_new_chat_message(self, chat: Chat, message: ChatMessage):
        if chat.hide_messages:
            return

        sender = chat.get_active_participant_by_name(message.sender_name)
        if sender is None:
            symbol = '❓'
        else:
            if sender.messages_hidden:
                return

            symbol = sender.symbol

        print(f'{symbol} {message.sender_name}: {message.content}')
