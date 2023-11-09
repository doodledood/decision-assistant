from chat.base import ChatRenderer, Chat, ChatMessage


class TerminalChatRenderer(ChatRenderer):
    def render_new_chat_message(self, chat: Chat, message: ChatMessage):
        if chat.hide_messages:
            return

        pretty_timestamp_with_date = message.timestamp.strftime('%m-%d-%Y %H:%M:%S')

        sender = chat.get_active_participant_by_name(message.sender_name)
        if sender is None:
            symbol = '❓'

            print(f'[{pretty_timestamp_with_date}] {symbol} {message.sender_name}: {message.content}')
        else:
            if sender.messages_hidden:
                return

            if chat.name is None:
                print(f'[{pretty_timestamp_with_date}] {str(sender)}: {message.content}')
            else:
                print(f'[{pretty_timestamp_with_date}] {chat.name} > {str(sender)}: {message.content}')
