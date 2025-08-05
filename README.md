# Candidate Recommendation Engine (SproutsAI Internship Assignment)

A Streamlit-based application that recommends the most relevant candidates for a job by combining **vector similarity search** with **LLM-powered evaluation** (RAG).

---

## 🚀 Live Demo

🔗 [Your Deployed App Link](https://sproutsai-assignment-sai-dileep-kumar-m.streamlit.app/)  

---

## 💡 Approach

1. **Input Collection**
   - Job description (text box)
   - Multiple resumes (PDF, DOCX, TXT)

2. **Resume Parsing**
   - Text extraction using appropriate file loaders

3. **Semantic Similarity Ranking**
   - Job description and resumes embedded with `OpenAIEmbeddings`
   - FAISS used to compute cosine similarity
   - Top 5 resumes selected

4. **LLM Evaluation**
   - `gpt-4o` processes each selected resume + job description
   - Returns a **concise summary** on candidate relevance and strengths

---

## ✅ Features

- Upload multiple resumes (PDF, DOCX, TXT)
- OpenAI cosine similarity for ranking
- GPT-4o summarization of top 5 candidates
- Secure API key via Streamlit secrets
- Clean & minimal UI using Streamlit with expandable recommendations

---

## 🔐 Assumptions

- Resumes are 1–2 pages (fit within LLM context window)
- Cosine similarity is based on normalized OpenAI embeddings
- Only top 5 resumes go through the extensive LLM summarization step

---

## 📦 Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
streamlit run main.py
