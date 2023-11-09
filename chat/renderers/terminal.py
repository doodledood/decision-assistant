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

        if chat.name is None:
            print(f'{symbol} {message.sender_name}: {message.content}')
        else:
            print(f'{chat.name} > {symbol} {message.sender_name}: {message.content}')
