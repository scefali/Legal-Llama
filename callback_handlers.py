"""Callback Handler streams to stdout on new llm token."""
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler


class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, on_token) -> None:
        self.on_token = on_token

    """Callback handler for streaming. Only works with LLMs that support streaming."""
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.on_token(token)
