# Enterprise Knowledge RAG

An enterprise document question-answering system built with Python, OpenAI, ChromaDB, FastAPI and Streamlit. The project implements a baseline vector-based Retrieval-Augmented Generation (RAG) pipeline, with particular emphasis on measurable retrieval quality, grounded answers, citation validation and systematic evaluation.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/API-FastAPI-009688) ![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange) ![Tests](https://img.shields.io/badge/Verified%20Tests-59%20passed-brightgreen)

## Overview

Enterprise teams often need to search operational runbooks, architecture guidelines, security policies and troubleshooting documents. This application retrieves relevant document passages and generates answers with numbered source citations. When sufficient evidence is unavailable, it abstains instead of inventing an answer.

The project is a local portfolio implementation of Naive RAG, not a production-ready enterprise deployment. Its evaluation framework makes retrieval and generation trade-offs visible and supports reproducible experiments.

## Key features

- PDF, Markdown and plain-text document ingestion with checksum-based duplicate detection.
- Recursive text chunking, metadata preservation and batched embedding generation.
- Persistent ChromaDB vector search with cosine similarity, configurable top-K and similarity threshold.
- Context-grounded generation using OpenAI, numbered citations and insufficient-context handling.
- Citation normalization for formats such as `[SOURCE 1]`, `[1]` and `[1, 2]`.
- FastAPI endpoints for querying, uploading, listing and deleting documents.
- Streamlit interface with source excerpts, similarity scores and execution metrics.
- Retrieval experiments, deterministic generation metrics, LLM-as-a-Judge and historical failure replay.

## Architecture

``` mermaid
flowchart TD
    D[Enterprise documents] --> L[Document loader]
    L --> C[Text chunker]
    C --> E[Embedding service]
    E --> V[(ChromaDB)]
    U[Streamlit UI] --> A[FastAPI]
    A --> R[RAG service]
    R --> Q[Query embedding]
    Q --> S[Vector retriever]
    V --> S
    S --> T{Relevant context?}
    T -->|No| N[Abstain]
    T -->|Yes| P[Prompt builder]
    P --> G[OpenAI answer generator]
    G --> B[Citation normalization and validation]
    B -->|Valid| O[Answer, sources and metrics]
    B -->|Invalid or missing| N
    O --> A
    N --> A
    R -.-> X[Execution trace]
    X -.-> Y[Evaluation framework]
```

## Technology stack

| Layer               | Technology                         |
|---------------------|------------------------------------|
| Language            | Python 3.12                        |
| Embeddings          | OpenAI `text-embedding-3-small`    |
| Generation          | OpenAI `gpt-4.1-mini`              |
| Semantic evaluator  | OpenAI `gpt-4.1`                   |
| Vector database     | ChromaDB, persistent cosine search |
| Document processing | PyPDF, LangChain text splitters    |
| Backend             | FastAPI, Pydantic                  |
| Frontend            | Streamlit, HTTPX                   |
| Testing             | pytest                             |

## Project structure

``` text
enterprise-knowledge-rag/
├── app/
│   ├── api/             # REST endpoints
│   ├── models/          # Domain models and execution traces
│   ├── ingestion/       # Loading and chunking
│   ├── embeddings/      # OpenAI embeddings
│   ├── vectorstore/     # ChromaDB integration
│   ├── retrieval/       # Similarity search
│   ├── generation/      # Prompts, generation and citations
│   ├── services/        # Ingestion and RAG orchestration
│   ├── scripts/         # Application utilities
│   ├── config.py
│   └── main.py
├── frontend/            # Streamlit application
├── data/
│   └── sample_documents/
├── evaluation/
│   ├── datasets/
│   └── reports/         # Generated benchmark reports
├── tests/
├── requirements.txt
└── .env.example
```

## Getting started

Prerequisites: Python 3.12 and an OpenAI API key with available API credits.

From the repository root:

``` bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Never commit API keys or the populated `.env` file.

### Configuration

``` dotenv
OPENAI_API_KEY=your_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_JUDGE_MODEL=gpt-4.1
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=2
MIN_SIMILARITY_SCORE=0.30
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHROMA_COLLECTION_NAME=enterprise_documents
API_HOST=0.0.0.0
API_PORT=8000
```

### Start the backend

``` bash
python -m uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

### Start the frontend

In another terminal, activate the virtual environment and run:

``` bash
python -m streamlit run frontend/app.py --server.port 8501
```

Open http://localhost:8501. Upload the sample documents using the sidebar before querying or running benchmarks. The backend persists indexed chunks locally in ChromaDB.

## API reference

| Method | Endpoint                          | Purpose                        |
|--------|-----------------------------------|--------------------------------|
| GET    | `/api/v1/health`                  | Health check                   |
| POST   | `/api/v1/documents`               | Upload and index a document    |
| GET    | `/api/v1/documents`               | List indexed documents         |
| DELETE | `/api/v1/documents/{document_id}` | Remove indexed document chunks |
| POST   | `/api/v1/query`                   | Ask a question                 |

``` bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How should database connection pools be monitored?","top_k":2}'
```

Illustrative answer after indexing the sample documents:

> Monitor active connections, pending acquisition requests and acquisition latency \[1\].

The response includes source metadata, excerpts, similarity scores and retrieval/generation latency and token metrics. Exact generated wording may vary.

## Sample knowledge base

The eight synthetic enterprise documents cover database operations, deployment guidelines, disaster recovery, incident management, Kafka troubleshooting, microservices architecture, security and Spark performance.

Example questions:

- How do I troubleshoot Kafka consumer lag?
- What is the recommended application rollback procedure?
- What authentication and resilience mechanisms should microservices implement?
- What are the documented RPO and RTO targets?
- What is the current consumer lag of the production payment consumer group? (Expected abstention: live telemetry is unavailable.)

## Evaluation methodology

The golden dataset contains 20 questions: 14 answerable and six unanswerable, with 58 expected facts. It covers direct retrieval, semantic paraphrases, multi-document questions, numerical questions, unavailable live metrics, out-of-domain queries and adversarial instructions.

### Retrieval evaluation

Metrics include Precision@K, Recall@K, MRR, hit rate, context precision and out-of-domain rejection. A sweep of 24 top-K/threshold combinations selected `K=2`, threshold `0.30`, preserving recall and rejection while reducing irrelevant context.

| Metric                    | Baseline K=5, 0.25 | Optimized K=2, 0.30 |
|---------------------------|-------------------:|--------------------:|
| Precision@K               |             22.35% |              55.88% |
| Recall@K                  |               100% |                100% |
| MRR                       |             0.9706 |              0.9706 |
| Context precision         |             41.76% |              73.53% |
| Mean retrieved chunks     |               2.70 |                1.40 |
| Irrelevant chunks (total) |                 35 |                   9 |

The selected configuration reduced retrieved chunks from 54 to 28 and irrelevant chunks by 74.29%, retaining all 19 relevant retrieved chunks in this experiment.

### Generation evaluation

Deterministic evaluation measures abstention, citation validity, guardrail intervention, tokens and latency. The LLM judge evaluates correctness, faithfulness, relevance, citation support and expected-fact coverage.

Historical benchmark results (before citation-parser enhancement):

| Metric                   | Baseline | Optimized |
|--------------------------|---------:|----------:|
| Abstention accuracy      |   85.00% |   100.00% |
| Answerable response rate |   78.57% |   100.00% |
| Citation validity        |   78.57% |   100.00% |
| Conditional correctness  |   95.45% |    92.86% |
| Conditional faithfulness |   97.73% |    98.21% |
| End-to-end correctness   |   75.00% |    92.86% |
| Micro fact coverage      |   74.14% |    91.38% |
| Expected facts covered   |    43/58 |     53/58 |
| Generation input tokens  |    9,086 |     6,311 |
| Total generation tokens  |   10,168 |     7,314 |
| Mean end-to-end latency  | 1,727 ms |  1,993 ms |

Input tokens decreased by 30.54% and total generation tokens by 28.07%. Mean latency increased in the observed run; lower token usage did not demonstrate lower latency. Conditional metrics exclude abstained answers, while end-to-end correctness counts false abstentions as zero.

### Engineering case study: citation false abstentions

Execution traces revealed that three baseline answers contained `[SOURCE 1]` rather than the expected `[1]`. The correct source was retrieved at rank one in all three cases, but the original citation parser rejected the format, causing false abstentions.

The fix introduced citation normalization, range validation and preservation of raw model output in execution traces.

| Verification                           | Observed result |
|----------------------------------------|----------------:|
| Citation parser tests                  |    20/20 passed |
| Complete regression suite              |    59/59 passed |
| Historical citation failures recovered |             3/3 |
| Remaining replay failures              |               0 |

Historical replay required no new LLM calls. It demonstrated structural citation recovery, not semantic correctness. The original benchmark reports remain unchanged. Generation quality improvements cannot be attributed solely to retrieval optimization because citation formatting and generation variability also affected the results.

## Running evaluation and tests

Run from the repository root with the virtual environment activated and the eight sample documents indexed for live benchmarks.

``` bash
# Retrieval evaluation and parameter sweep
python -m evaluation.run_retrieval_evaluation
python -m evaluation.run_retrieval_experiment

# Generation benchmarks (uses paid API calls)
python -m evaluation.run_generation_evaluation --top-k 5 --threshold 0.25
python -m evaluation.run_generation_evaluation --top-k 2 --threshold 0.30

# Compare generated reports
python -m evaluation.compare_generation_reports \
  --baseline "$(ls -t evaluation/reports/generation_k5_threshold0.25_*.json | head -n 1)" \
  --optimized "$(ls -t evaluation/reports/generation_k2_threshold0.3_*.json | head -n 1)"

# Offline historical replay (use an original pre-fix baseline report)
python -m evaluation.replay_citation_failures --report path/to/original_baseline_report.json

# Regression suite
python -m pytest tests/ -v
```

Replace the replay report path with the actual preserved historical report. New benchmark runs use the current code and may not reproduce historical results exactly. Retrieval and generation evaluation can incur OpenAI API charges. Reports may contain full document passages and model responses; review them before publishing.

## Design decisions and limitations

- Direct OpenAI SDK calls keep generation and embedding orchestration explicit.
- Similarity thresholding and top-K tuning balance context coverage and noise; the chosen threshold is dataset-specific.
- Citation validation checks syntax and source IDs, not runtime claim-level entailment.
- The small synthetic dataset was reused for tuning and evaluation; results are development benchmarks, not independent production validation. Each sample document fits in one chunk, so long-document chunking quality is not established by these benchmarks.
- LLM judge scores may be inconsistent. Formal judge calibration was deferred.
- The application uses a shared local knowledge base without enterprise authentication, document-level authorization or live operational telemetry.
- Prompt instructions are not a complete defense against prompt injection.

## Future work

- Validate the proposed generation prompt improvements for answer completeness and groundedness.
- Expand the evaluation dataset and introduce held-out tests and judge calibration.
- Add broader application integration tests, Docker Compose and CI automation.
- Explore hybrid retrieval, reranking, Corrective RAG and other advanced RAG architectures.
- Introduce access control, observability and production deployment safeguards.

## Author

[GitHub — badalb](https://github.com/badalb)
