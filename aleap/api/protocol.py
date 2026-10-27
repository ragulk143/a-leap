from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatCompletionMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-aleap"
    object: str = "chat.completion"
    created: int = 2026
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16})
