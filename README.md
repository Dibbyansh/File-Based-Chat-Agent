# 📄 File Q&A Bot

A simple AI-powered File Question Answering system built using a **Retrieval-Augmented Generation (RAG)** approach.  
It answers questions strictly based on the content of a given file.

---

## 🚀 Features

- 📂 Load and process any `.txt` file  
- ✂️ Automatic text chunking  
- 🔍 Semantic search using embeddings  
- 🧠 Context-based answers using LLM  
- ❌ No hallucination (answers only from file)  

---

## 🧠 How it works

```text
User Input (File + Question)
   ↓
Text Loader (reads file)
   ↓
Text Splitter (creates chunks)
   ↓
Embeddings Model (vector conversion)
   ↓
Chroma DB (stores vectors)
   ↓
Retriever (fetches relevant chunks)
   ↓
LLM (OpenRouter)
   ↓
Final Answer (based only on file)
```

## 📌 Example

```text
Enter file path: yoga.txt

You: From where the word 'yoga' comes from?

Assistant:
The Sanskrit word 'yoga' comes from the word 'yuj' which means, 'to unite'.
```

## 🛠 Tech Stack

- **Python**
- **LangChain** – RAG pipeline
- **OpenRouter** – LLM API  
- **Chroma DB** – Vector database  
- **HuggingFace Embeddings** – `all-MiniLM-L6-v2`  

---
