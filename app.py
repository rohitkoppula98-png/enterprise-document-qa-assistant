import streamlit as st
st.set_page_config(
    page_title="Enterprise Document Q&A",
    page_icon="📚",
    layout="wide"
)
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

st.title("📚 Enterprise Document Q&A Assistant")

st.markdown(
    "### Your AI-powered document intelligence assistant"
)

st.write(
    "Upload one or more PDF documents, ask questions, "
    "and receive AI-generated answers with source references."
)

st.divider()

uploaded_files = st.file_uploader(
    "Choose a PDF document",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    pages = []

    for uploaded_file in uploaded_files:

        pdf_reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages.append({
                    "page": page_number,
                    "filename": uploaded_file.name,
                    "text": page_text.strip()
                })

    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")

    # Split each page into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    chunk_pages = []
    chunk_filenames = []

    for page in pages:
        page_chunks = text_splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append(chunk)
            chunk_pages.append(page["page"])
            chunk_filenames.append(page["filename"])

    st.write(f"Document split into {len(chunks)} chunks.")

    if chunks:

        # Create embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks).tolist()

        # Create ChromaDB collection
        client = chromadb.Client()
        collection = client.get_or_create_collection(name="documents")
        existing_ids = collection.get()["ids"]

        if existing_ids:
            collection.delete(ids=existing_ids)

        chunk_metadatas = [
            {"page": page_number, "filename": filename}
            for page_number, filename in zip(chunk_pages, chunk_filenames)
        ]

        # Store chunks
        collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadatas
        )
        st.success("Document stored successfully!")

        # Ask a question
        question = st.text_input(
            "Ask a question about your PDF"
        )

        if question:

            # Retrieve relevant chunks
            query_embedding = model.encode(question).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(2, len(chunks))
            )

            relevant_chunks = results["documents"][0]
            relevant_metadatas = results["metadatas"][0]

            st.subheader("Retrieved information")

            for chunk, metadata in zip(relevant_chunks, relevant_metadatas):
                st.write(chunk)
                st.caption(f"Source: {metadata['filename']} — Page {metadata['page']}")

            # Generate answer using Claude
            if api_key:

                client_claude = anthropic.Anthropic(
                    api_key=api_key
                )

                context = "\n\n".join(relevant_chunks)

                with st.spinner("Claude is generating your answer..."):

                    response = client_claude.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=500,
                        messages=[
                            {
                                "role": "user",
                                "content": f"""
                                Answer the question using only the
                                document context below.

                                If the answer is not present in the
                                context, say you cannot find it in
                                the document.

                                Document context:
                                {context}

                                Question:
                                {question}
                                """
                            }
                        ]
                    )

                st.subheader("Claude's Answer")
                st.write(response.content[0].text)

            else:
                st.error(
                    "API key not found. Check your .env file."
                )

    else:
        st.warning("No readable text found in this PDF.")