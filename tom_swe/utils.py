import json
from typing import Generic, Type, TypeVar

import json_repair
from pydantic import BaseModel

OutputType = TypeVar("OutputType", bound=object)
T = TypeVar("T", bound=BaseModel)


class OutputParser(BaseModel, Generic[OutputType]):
    def parse(self, result: str) -> OutputType:
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        raise NotImplementedError


class PydanticOutputParser(OutputParser[T], Generic[T]):
    pydantic_object: Type[T]

    def parse(self, result: str) -> T:
        json_result = json_repair.loads(result)
        assert isinstance(json_result, dict)
        if "properties" in json_result:
            return self.pydantic_object.model_validate_json(json.dumps(json_result["properties"]))
        else:
            parsed_result = self.pydantic_object.model_validate_json(result)
            return parsed_result

    def get_format_instructions(self) -> str:
        return json.dumps(self.pydantic_object.model_json_schema())


def split_text_for_embedding(text: str, max_tokens: int = 8191) -> list[str]:
    """
    Split text into chunks that stay within token limits for embedding models.
    Uses tiktoken for accurate token counting when available.
    """
    safe_max_tokens = min(max_tokens - 300, 7800)

    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text, disallowed_special=())

        if len(tokens) <= safe_max_tokens:
            return [text]

        # Split into chunks
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + safe_max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end

        return chunks

    except ImportError:
        # Fallback to character-based estimation
        max_chars = int(safe_max_tokens * 2.8)
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end

        return chunks
