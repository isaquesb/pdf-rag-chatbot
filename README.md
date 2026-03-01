# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain that lets you ask questions about a collection of PDFs.

## How it works

1. **Load** — PDFs from `./docs/` are loaded page-by-page via `PyPDFLoader`
2. **Chunk** — Pages are split into 750-token overlapping chunks with `RecursiveCharacterTextSplitter`
3. **Embed** — Chunks are embedded with `OpenAIEmbeddings` and stored in a local Chroma vector store (`./data/chroma_db/`)
4. **Retrieve & Answer** — At query time, the top-3 most relevant chunks are retrieved and passed to `gpt-3.5-turbo` via an LCEL chain

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

```bash
pip install langchain-core langchain-classic langchain-community langchain-openai langchain-chroma langchain-text-splitters python-dotenv pypdf jupyter
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Add your PDFs to the `./docs/` directory.

## Usage

Open `notebook.ipynb` in Jupyter and run the cells in order:

| Cells | What happens |
|-------|-------------|
| 1–2   | Imports and environment setup |
| 3     | PDF loading |
| 4     | Text splitting |
| 5     | Build & persist Chroma vector store |
| 6     | Build the RAG chain |
| 7     | Interactive Q&A prompt |

Run cells 1–5 once to build the vector store. On subsequent runs you can skip to cell 6.

## Stack

| Package | Version | Role |
|---------|---------|------|
| `langchain-classic` | 1.0.1 | `create_retrieval_chain`, `create_stuff_documents_chain` |
| `langchain-core` | 1.2.16 | `ChatPromptTemplate` |
| `langchain-community` | 0.4.1 | `PyPDFLoader` |
| `langchain-openai` | 1.1.10 | `OpenAIEmbeddings`, `ChatOpenAI` |
| `langchain-chroma` | 1.1.0 | Chroma vector store |
| `langchain-text-splitters` | 1.1.1 | `RecursiveCharacterTextSplitter` |
| `python-dotenv` | 1.2.1 | `.env` loading |
