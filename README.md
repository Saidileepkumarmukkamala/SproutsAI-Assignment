# Candidate Recommendation Engine (SproutsAI Internship Assignment)

A Streamlit-based application that recommends the most relevant candidates for a job by combining **vector similarity search** with **LLM-powered evaluation** (RAG).

---

## 🚀 Live Demo

🔗 [Your Deployed App Link](https://sproutsai-assignment-sai-dileep-kumar-m.streamlit.app/)  

---

## 💡 Approach

1. **Input Collection**
   - Accepts a job description via text box.
   - Accepts multiple resumes via file upload (PDF, DOCX, or TXT).

2. **Text Parsing**
   - Extracts text using format-appropriate loaders.

3. **Vector Similarity (FAISS + OpenAI Embeddings)**
   - Converts both job description and resumes into embeddings.
   - Uses `FAISS.similarity_search_with_score()` to rank resumes by cosine similarity.
   - Top 5 most relevant resumes are selected for further evaluation.

4. **LLM Evaluation (RAG)**
   - Uses OpenAI's `gpt-4o` model via LangChain’s `RetrievalQA`.
   - A custom prompt is used to generate a **concise summary** explaining each candidate’s fit.

---

## ✅ Features

- ✅ Cosine similarity-based candidate ranking
- ✅ LLM-generated explanations per candidate
- ✅ Top 5 candidates automatically selected for deeper analysis
- ✅ Upload resumes in common formats: `.pdf`, `.docx`, `.txt`
- ✅ Secure API key handling via Streamlit secrets

---

## 🔐 Assumptions

- Resumes are typically 1–2 pages long and can fit into the LLM context window.
- Cosine similarity is computed using OpenAI embeddings and FAISS.
- LLM summarization is only applied to the top 5 resumes (for speed and cost efficiency).
- API keys are not hardcoded but managed securely using `.streamlit/secrets.toml` (locally) or Streamlit Cloud secrets.

---

## 📦 Setup Instructions

1. **Install requirements:**

```bash
pip install -r requirements.txt
streamlit run main.py
