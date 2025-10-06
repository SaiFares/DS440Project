# 🧠 RAG-Based Healthcare Chatbot  
**Course:** DS440 – Advanced Data Systems  
**Project Title:** Retrieval-Augmented Generation (RAG) for Healthcare Question Answering  
**Author:** Saif, Abdul  
**Institution:** Penn State University  

---

## 📘 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system for healthcare question answering using *The Gale Encyclopedia of Medicine* as the core knowledge base.  
The chatbot combines **document retrieval** and **large language model generation** to provide reliable, context-aware, and verifiable answers to user medical queries.

The purpose of this project is to demonstrate how **RAG architectures** can enhance factual accuracy and explainability in medical conversational systems by grounding responses in trusted data sources.

---

## 🧩 System Architecture
### 1. Document Preprocessing  
- Articles from *The Gale Encyclopedia of Medicine* are converted into clean text.  
- Text is segmented into chunks for embedding and retrieval.  

### 2. Vector Database Creation  
- Sentence embeddings are generated using transformer models such as  
  `sentence-transformers/all-MiniLM-L6-v2`.  
- Embeddings are stored in a **vector database** (FAISS / Chroma / Pinecone).  

### 3. Retrieval-Augmented Generation  
- On each user query:
  1. The retriever fetches top-k relevant text chunks using cosine similarity.  
  2. Retrieved content is passed to an LLM (e.g., GPT-based model) for contextual generation.  
  3. The LLM produces a natural-language, reference-based answer.  

---

## ⚙️ Tech Stack
| Component | Technology Used |
|------------|-----------------|
| **Language** | Python 3.10+ |
| **Embeddings** | SentenceTransformers |
| **Vector Store** | FAISS / Chroma |
| **LLM Backend** | OpenAI API / Ollama (local) |
| **Frameworks** | LangChain / LlamaIndex |
| **Data Source** | The Gale Encyclopedia of Medicine |
| **Frontend (optional)** | Streamlit or Gradio |

---

## 🚀 How to Run
1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/rag-healthcare-bot.git
   cd rag-healthcare-bot
