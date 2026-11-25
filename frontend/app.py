import httpx
import streamlit as st

from api_client import APIClient


st.set_page_config(
    page_title="Enterprise Knowledge RAG",
    page_icon="📚",
    layout="wide"
)


client = APIClient()


if "messages" not in st.session_state:

    st.session_state.messages = []


def display_error(exc):

    if isinstance(exc, httpx.HTTPStatusError):

        try:

            detail = exc.response.json().get(
                "detail",
                "Request failed"
            )

        except Exception:

            detail = "Request failed"

        st.error(
            f"HTTP {exc.response.status_code}: {detail}"
        )

    else:

        st.error(f"Unable to complete request: {exc}")


def render_response(response):

    st.markdown(response["answer"])

    sources = response.get("sources", [])

    if sources:

        with st.expander(
            f"Source citations ({len(sources)})"
        ):

            for source in sources:

                st.markdown(
                    f"**[{source['citation_id']}] "
                    f"{source['filename']}**"
                )

                st.caption(
                    f"Similarity: "
                    f"{source['similarity_score']:.4f}"
                )

                if source.get("page_number") is not None:

                    st.caption(
                        f"Page: {source['page_number']}"
                    )

                st.text(source["excerpt"])

                st.divider()

    metrics = response.get("metrics", {})

    if metrics:

        with st.expander("Performance metrics"):

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Retrieved chunks",
                metrics["retrieved_chunks"]
            )

            col2.metric(
                "Retrieval",
                f"{metrics['retrieval_latency_ms']:.0f} ms"
            )

            col3.metric(
                "Generation",
                f"{metrics['generation_latency_ms']:.0f} ms"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Total latency",
                f"{metrics['total_latency_ms']:.0f} ms"
            )

            col5.metric(
                "Input tokens",
                metrics.get("input_tokens") or 0
            )

            col6.metric(
                "Output tokens",
                metrics.get("output_tokens") or 0
            )


# ------------------------------------------------
# Sidebar
# ------------------------------------------------

with st.sidebar:

    st.header("Configuration")

    try:

        health = client.health()

        if health.get("status") == "UP":

            st.success("Backend connected")

    except httpx.HTTPError:

        st.error("Backend unavailable")

        st.stop()

    top_k = st.slider(
        "Top-K retrieval",
        min_value=1,
        max_value=20,
        value=5
    )

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.header("Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "txt", "md"]
    )

    if uploaded_file is not None:

        if st.button(
            "Index document",
            use_container_width=True
        ):

            try:

                with st.spinner("Indexing document..."):

                    result = client.upload_document(
                        filename=uploaded_file.name,
                        content=uploaded_file.getvalue(),
                        content_type=(
                            uploaded_file.type
                            or "application/octet-stream"
                        )
                    )

                if result["status"] == "INDEXED":

                    st.success(
                        f"Indexed {result['chunks']} chunks"
                    )

                else:

                    st.info(
                        result.get("reason")
                        or "Document already indexed"
                    )

            except httpx.HTTPError as exc:

                display_error(exc)

    try:

        documents = client.list_documents()

        st.metric(
            "Indexed documents",
            len(documents)
        )

        st.metric(
            "Stored chunks",
            sum(
                document["chunk_count"]
                for document in documents
            )
        )

        with st.expander("Manage documents"):

            for document in documents:

                st.markdown(
                    f"**{document['filename']}**"
                )

                st.caption(
                    f"Chunks: {document['chunk_count']}"
                )

                if st.button(
                    "Delete",
                    key=document["document_id"]
                ):

                    client.delete_document(
                        document["document_id"]
                    )

                    st.rerun()

                st.divider()

    except httpx.HTTPError as exc:

        display_error(exc)


# ------------------------------------------------
# Main application
# ------------------------------------------------

st.title("Enterprise Knowledge RAG")

st.caption(
    "Ask questions about your enterprise engineering "
    "documents and receive grounded answers with citations."
)


if not st.session_state.messages:

    st.info(
        "Try asking about Kafka consumer lag, Spark data skew, "
        "database connection pools or disaster recovery."
    )


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":

            st.markdown(message["content"])

        else:

            render_response(message["response"])


question = st.chat_input(
    "Ask a question about your knowledge base"
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Retrieving evidence and generating answer..."
            ):

                response = client.query(
                    question=question,
                    top_k=top_k
                )

            render_response(response)

            st.session_state.messages.append({
                "role": "assistant",
                "response": response
            })

        except httpx.HTTPError as exc:

            display_error(exc)