from pydantic import BaseModel


class GenerationResult(BaseModel):

    answer: str

    input_tokens: int | None = None

    output_tokens: int | None = None