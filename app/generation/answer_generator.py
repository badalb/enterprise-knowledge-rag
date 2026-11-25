from openai import OpenAI

from app.config import get_settings
from app.models.generation import GenerationResult


class AnswerGenerator:

    def __init__(self):

        settings = get_settings()

        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

        self.model = settings.openai_chat_model

    def generate(
        self,
        messages: list[dict[str, str]]
    ) -> GenerationResult:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise ValueError(
                "LLM returned an empty response"
            )

        usage = response.usage

        return GenerationResult(
            answer=answer.strip(),

            input_tokens=(
                usage.prompt_tokens
                if usage else None
            ),

            output_tokens=(
                usage.completion_tokens
                if usage else None
            )
        )