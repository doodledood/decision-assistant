from typing import Type, Optional


from chat.base import ActiveChatParticipant, Chat, TOutputSchema
from chat.errors import NoMessagesInChatError
from chat.utils import fix_invalid_json, json_string_to_pydantic


class JSONOutputParserChatParticipant(ActiveChatParticipant):
    output_schema: Type[TOutputSchema]
    output: Optional[TOutputSchema] = None

    def __init__(self,
                 output_schema: Type[TOutputSchema],
                 name: str = 'JSON Output Parser',
                 role: str = 'JSON Output Parser'
                 ):
        super().__init__(name=name, role=role)

        self.output_schema = output_schema

    def respond_to_chat(self, chat: Chat) -> str:
        messages = chat.get_messages()
        if len(messages) == 0:
            raise NoMessagesInChatError()

        last_message = messages[-1]

        try:
            json_string = last_message.content[last_message.content.index('{'):last_message.content.rindex('}') + 1]
            json_string = fix_invalid_json(json_string)
            self.output = model = json_string_to_pydantic(json_string, self.output_schema)

            return f'{model.model_dump_json()} TERMINATE'
        except Exception as e:
            return f'I could not parse the JSON. This was the error: {e}'


