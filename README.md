
# Enterprise Document Q&A Assistant

An AI-powered application that allows users to upload PDF documents, ask questions, and receive answers based on the content of those documents.

## Features

- Upload multiple PDF documents.
- Extract text from PDF files.
- Split documents into smaller chunks.
- Generate text embeddings using Sentence Transformers.
- Store and retrieve document chunks using ChromaDB.
- Use semantic search to find relevant information.
- Generate answers using Anthropic Claude.
- Display source document names and page numbers.
- Avoid answering questions when the information is not found in the retrieved context.

## Technologies Used

- Python
- Streamlit
- PyPDF
- LangChain Text Splitters
- Sentence Transformers
- ChromaDB
- Anthropic Claude API
- Python dotenv

## How It Works

1. The user uploads one or more PDF files.
2. The application extracts text from each PDF.
3. The text is divided into smaller chunks.
4. Sentence Transformers converts the chunks into embeddings.
5. ChromaDB stores the chunks and their metadata.
6. The user's question is converted into an embedding.
7. Semantic search retrieves the most relevant chunks.
8. Claude generates an answer using the retrieved document context.
9. The application displays the answer and its source references.

## Setup Instructions

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies.
4. Create a `.env` file in the project folder.
5. Add your Anthropic API key:

   `ANTHROPIC_API_KEY=your_api_key_here`

6. Run the application:

   `streamlit run app.py`

## Important Notes

- An Anthropic API key is required.
- API usage may incur charges.
- The current version uses in-memory ChromaDB storage, so documents need to be uploaded again after the application restarts.
