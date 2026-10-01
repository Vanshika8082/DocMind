# 🧠 DocMind — AI Document Assistant

DocMind is an AI-powered document question-answering application built using **Retrieval-Augmented Generation (RAG)** concepts.

The project allows users to provide documents and interact with their content using natural-language questions. It uses document loading, text splitting, embeddings/vector search, and an LLM to retrieve relevant information and generate answers.

## 🚀 Features

- 📄 Upload and process documents
- ✂️ Split documents into smaller chunks
- 🔎 Retrieve relevant document content
- 🧠 Generate answers using an LLM
- 💬 Ask questions about uploaded documents
- ⚡ Fast AI inference
- 🔐 API keys managed through environment variables

## 🧩 How It Works

DocMind follows a basic **RAG (Retrieval-Augmented Generation)** pipeline:

```text
             Document
                 │
                 ▼
        ┌─────────────────┐
        │ Document Loader │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Text Splitter  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │    Embeddings   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Vector Store  │
        └────────┬────────┘
                 │
           User Question
                 │
                 ▼
        ┌─────────────────┐
        │    Retrieval    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │       LLM       │
        └────────┬────────┘
                 │
                 ▼
              Answer
```

## 🛠️ Technologies

- **Python**
- **LangChain**
- **RAG (Retrieval-Augmented Generation)**
- **Vector Database**
- **Document Loaders**
- **Text Splitters**
- **Embeddings**
- **Groq API / LLM**
- **Git & GitHub**

## 📂 Project Structure

```text
DocMind/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
│
├── [application files]
├── [source files]
└── [other project files]
```

> The exact structure may vary depending on the implementation.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vanshika8082/DocMind.git
cd DocMind
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your `.env` file to GitHub.

Your `.gitignore` should contain:

```gitignore
.env
venv/
__pycache__/
```

## ▶️ Running the Project

After installing the dependencies and configuring your API key, run the application's main file.

For example:

```bash
python app.py
```

Use the actual entry-point file from your project if it has a different name.

