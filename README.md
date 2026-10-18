# 📚 AI Document Chat — RAG Application

An AI-powered document chat application that allows users to upload PDF documents and ask questions about their content.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the uploaded document and provide answers using a Large Language Model.

## 🚀 Features

- 📄 Upload PDF documents
- ✂️ Split documents into smaller chunks
- 🔢 Generate embeddings using Hugging Face
- 🗄️ Store document embeddings in ChromaDB
- 🔍 Retrieve relevant chunks using MMR search
- 🤖 Generate answers using Groq LLM
- 💬 Chat with documents through a Streamlit interface
- 📚 Support different documents without manually rebuilding the application
- 🔐 Environment variables for securely storing API keys

## 🧠 How It Works

The application follows a RAG pipeline:

```text
                    PDF Document
                         │
                         ▼
                  Document Loader
                         │
                         ▼
                  Text Chunking
                         │
                         ▼
                 Embedding Model
                         │
                         ▼
                     ChromaDB
                  Vector Database
                         │
                         │
User Question ───────────┘
       │
       ▼
    Retriever
       │
       ▼
 Relevant Document Chunks
       │
       ▼
    LLM (Groq)
       │
       ▼
     Answer
```

### RAG Pipeline

1. **Document Loading**

   PDF files are loaded using `PyPDFLoader`.

2. **Text Splitting**

   The document is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

3. **Embedding Generation**

   Each chunk is converted into a vector using:

   `sentence-transformers/all-MiniLM-L6-v2`

4. **Vector Storage**

   The embeddings are stored in **ChromaDB**.

5. **Retrieval**

   When the user asks a question, the application searches the vector database for relevant chunks using **MMR (Maximal Marginal Relevance)**.

6. **Generation**

   The retrieved context is provided to the Groq LLM, which generates the final answer.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| LangChain | RAG application framework |
| Hugging Face | Embedding model |
| ChromaDB | Vector database |
| Groq | LLM inference |
| Streamlit | Web interface |
| PyPDF | PDF processing |

## 📁 Project Structure

```text
rag-document-chat/
│
├── app.py
├── main.py
├── create_database.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── uploads/
│
└── chroma_db/
```

### Important Files

**`app.py`**

Main Streamlit application. It provides the user interface for uploading documents and chatting with them.

**`create_database.py`**

Creates the document embeddings and stores them in ChromaDB.

**`main.py`**

Contains the basic command-line RAG implementation.

**`requirements.txt`**

Contains the Python dependencies required to run the project.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

```bash
cd YOUR_REPOSITORY_NAME
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

**Never commit your `.env` file to GitHub.**

The repository contains `.env.example` as a template.

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can open the application in your browser.

## 💬 Using the Application

1. Start the Streamlit application.
2. Upload a PDF document.
3. The application processes the document.
4. The document is split into chunks.
5. Embeddings are generated.
6. The embeddings are stored in ChromaDB.
7. Ask questions about the uploaded document.
8. The RAG system retrieves relevant information and generates an answer.

## 🔍 Retrieval Configuration

The application uses MMR retrieval:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

- `k` — number of final documents returned
- `fetch_k` — number of documents considered before selecting the final results
- `lambda_mult` — controls the balance between relevance and diversity

## 📌 Important

The application is designed to answer questions using the retrieved document context.

If the required information is not available in the document, the system is instructed to respond:

> "I could not find the answer in the document."

## 🔐 Security

Do not upload or commit sensitive information such as:

- API keys
- `.env` files
- Personal documents
- Private credentials

The following are ignored by Git:

```text
.env
chroma_db/
uploads/
.venv/
__pycache__/
```

## 📈 Future Improvements

Possible improvements include:

- Support for multiple file formats
- Source/page citations in answers
- Conversation memory
- Multiple document collections
- Document management
- Improved retrieval strategies
- Reranking retrieved documents
- Streaming LLM responses
- Authentication
- Deployment to the cloud

## 👨‍💻 Author

**Vallabhaneni Chenchaiah**

This project was created as part of my learning journey in **Generative AI, RAG, and AI Engineering**.