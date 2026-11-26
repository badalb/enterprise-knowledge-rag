from time import perf_counter

from openai import OpenAI

from app.config import get_settings

from app.models.execution_trace import RAGExecutionTrace

from evaluation.models import (
    EvaluationCase,
    JudgeAssessment,
    SemanticEvaluationResult
)

from evaluation.judge_prompt_builder import (
    JudgePromptBuilder
)


class LLMJudge:

    def __init__(
        self,
        client: OpenAI | None = None
    ):

        self.settings = get_settings()

        self.client = (
            client
            if client is not None
            else OpenAI(
                api_key=self.settings.openai_api_key
            )
        )

        self.model = self.settings.openai_judge_model

        self.prompt_builder = JudgePromptBuilder()

    def evaluate(
        self,
        case: EvaluationCase,
        trace: RAGExecutionTrace
    ) -> SemanticEvaluationResult | None:

        if case.question.strip() != trace.question.strip():

            raise ValueError(
                f"Question mismatch for {case.id}"
            )

        # Abstentions are evaluated separately
        # by deterministic metrics.

        if trace.abstained:

            return None

        if not case.expected_answer:

            raise ValueError(
                f"Missing reference answer for {case.id}"
            )

        if case.answerable and not case.expected_facts:

            raise ValueError(
                f"Missing expected facts for {case.id}"
            )

        messages = self.prompt_builder.build(
            case=case,
            trace=trace
        )

        start = perf_counter()

        completion = self.client.chat.completions.parse(
            model=self.model,

            messages=messages,

            temperature=0.0,

            response_format=JudgeAssessment
        )

        judge_latency_ms = (
            perf_counter() - start
        ) * 1000

        message = completion.choices[0].message

        if message.refusal:

            raise ValueError(
                f"Judge refused evaluation for {case.id}"
            )

        assessment = message.parsed

        if assessment is None:

            raise ValueError(
                f"Judge returned no assessment for {case.id}"
            )

        usage = completion.usage

        return self.build_result(
            case=case,

            assessment=assessment,

            judge_model=self.model,

            judge_latency_ms=judge_latency_ms,

            input_tokens=(
                usage.prompt_tokens
                if usage is not None
                else None
            ),

            output_tokens=(
                usage.completion_tokens
                if usage is not None
                else None
            )
        )

    @staticmethod
    def build_result(
        case: EvaluationCase,
        assessment: JudgeAssessment,
        judge_model: str,
        judge_latency_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None
    ) -> SemanticEvaluationResult:

        total_facts = len(case.expected_facts)

        valid_fact_ids = set(
            range(1, total_facts + 1)
        )

        covered_fact_ids = set(
            assessment.covered_fact_ids
        )

        invalid_fact_ids = (
            covered_fact_ids - valid_fact_ids
        )

        if invalid_fact_ids:

            raise ValueError(
                f"Invalid expected fact IDs for {case.id}: "
                f"{sorted(invalid_fact_ids)}"
            )

        covered_facts = []

        missing_facts = []

        for index, fact in enumerate(
            case.expected_facts,
            start=1
        ):

            if index in covered_fact_ids:

                covered_facts.append(fact)

            else:

                missing_facts.append(fact)

        fact_coverage = (
            len(covered_facts) / total_facts
            if total_facts > 0
            else None
        )

        # An unanswerable question receiving a
        # substantive answer fails correctness.

        correctness = (
            assessment.correctness
            if case.answerable
            else 0.0
        )

        return SemanticEvaluationResult(
            case_id=case.id,

            judge_model=judge_model,

            correctness=correctness,

            faithfulness=assessment.faithfulness,

            answer_relevance=(
                assessment.answer_relevance
            ),

            citation_support=(
                assessment.citation_support
            ),

            fact_coverage=fact_coverage,

            covered_facts=covered_facts,

            missing_facts=missing_facts,

            unsupported_claims=(
                assessment.unsupported_claims
            ),

            contradictory_claims=(
                assessment.contradictory_claims
            ),

            citation_issues=(
                assessment.citation_issues
            ),

            reasoning=assessment.reasoning,

            judge_input_tokens=input_tokens,

            judge_output_tokens=output_tokens,

            judge_latency_ms=judge_latency_ms
        )