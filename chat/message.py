from pydantic import BaseModel


class ChatMessage(BaseModel):
    id: int
    sender_name: str
    content: str
