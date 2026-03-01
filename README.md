# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain that lets you ask questions about a collection of PDFs.

Two notebooks are provided, each with a different retrieval strategy. Choose the one that best fits your use case.

## How it works

### `notebook.ipynb` — Standard RAG

1. **Load** — PDFs from `./docs/` are loaded page-by-page via `PyPDFLoader`
2. **Chunk** — Pages are split into 750-token overlapping chunks with `RecursiveCharacterTextSplitter`
3. **Embed** — Chunks are embedded with `OpenAIEmbeddings` and stored in a local Chroma vector store (`./data/chroma_db/`)
4. **Retrieve & Answer** — At query time, the top-3 most relevant chunks are retrieved and passed to `gpt-3.5-turbo`

### `notebook_pdr.ipynb` — Parent Document Retriever (PDR)

1. **Load** — Same as above
2. **Split** — Each document is first split into large *parent* chunks (2000 tokens), then each parent into small *child* chunks (400 tokens)
3. **Embed** — Only the child chunks are embedded and stored in Chroma (`./data/chroma_db_pdr/`)
4. **Store** — Parent chunks are persisted separately in a pickle-based doc store (`./data/pdr_docstore.pkl`)
5. **Retrieve & Answer** — Semantic search finds the most relevant child chunks; their parent docs (richer context) are passed to `gpt-3.5-turbo`

## When to use each version

| | `notebook.ipynb` | `notebook_pdr.ipynb` |
|---|---|---|
| **Best for** | Short/simple documents, quick prototyping | Long documents with dense information |
| **Context sent to LLM** | 750 tokens × k | 2000 tokens × k |
| **Embedding precision** | Moderate (750-token chunks) | High (400-token child chunks) |
| **Setup complexity** | Simple | Moderate |
| **Persistence** | Chroma only | Chroma + pickle doc store |
| **`max_tokens` (LLM)** | 200 | 500 |

**Use `notebook.ipynb`** when your documents are short, you want a minimal setup, or you are experimenting with the pipeline.

**Use `notebook_pdr.ipynb`** when your documents are long and information-dense, and you need richer context in the LLM response without sacrificing embedding quality.

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

### Standard RAG — `notebook.ipynb`

Open the notebook in Jupyter and run the cells in order:

| Cells | What happens |
|-------|-------------|
| 1–2   | Imports and environment setup |
| 3     | PDF loading |
| 4     | Text splitting |
| 5     | Build & persist Chroma vector store |
| 6     | Build the RAG chain |
| 7     | Interactive Q&A prompt |

Run cells 1–5 once to build the vector store. On subsequent runs you can skip to cell 6.

### Parent Document Retriever — `notebook_pdr.ipynb`

Open the notebook in Jupyter and run the cells in order:

| Cells | What happens |
|-------|-------------|
| 1–2   | Imports and environment setup |
| 3     | PDF loading |
| 4     | Parent and child text splitters |
| 5     | `PickleDocStore` class and instantiation |
| 6     | Build index (parents → pickle, children → Chroma); skipped on reruns |
| 7     | `PDRetriever` class and RAG chain |
| 8     | Interactive Q&A prompt |

Run cells 1–6 once to build the index. On subsequent runs cells 1–7 load the persisted index automatically — no rebuilding needed.

To force a full rebuild, delete `./data/pdr_docstore.pkl` and `./data/chroma_db_pdr/` before running.

## Stack

| Package | Version | Role |
|---------|---------|------|
| `langchain-classic` | 1.0.1 | `create_retrieval_chain`, `create_stuff_documents_chain` |
| `langchain-core` | 1.2.16 | `ChatPromptTemplate`, `BaseRetriever`, `Document` |
| `langchain-community` | 0.4.1 | `PyPDFLoader` |
| `langchain-openai` | 1.1.10 | `OpenAIEmbeddings`, `ChatOpenAI` |
| `langchain-chroma` | 1.1.0 | Chroma vector store |
| `langchain-text-splitters` | 1.1.1 | `RecursiveCharacterTextSplitter` |
| `python-dotenv` | 1.2.1 | `.env` loading |
| `pypdf` | 6.7.4 | PDF parsing (used by `PyPDFLoader`) |
