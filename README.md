# Candidate Recommendation Engine (SproutsAI Internship Assignment)

This is a Streamlit-based web application that recommends the most relevant candidates for a job description using LLM-powered semantic search and summarization (RAG).

## 🚀 Live App

🔗 [Click here to view the app](https://your-username-your-app-name.streamlit.app/)  
---

## 💡 Approach

1. **Input Collection:**
   - Accepts a job description via text input.
   - Accepts resumes via file upload (PDF, DOCX, or TXT).

2. **Document Parsing:**
   - Extracts text from each resume using appropriate loaders.

3. **Embedding & Retrieval (RAG):**
   - Embeds resume content using `OpenAIEmbeddings`.
   - Stores embeddings in a FAISS vector store.
   - For each resume, a Retrieval-Augmented Generation (RAG) chain is used to evaluate its relevance to the job description.

4. **LLM Evaluation:**
   - Uses `gpt-4o` (or latest OpenAI model).
   - Prompts the model to return a **concise, clear summary** explaining why the candidate is (or isn’t) a good fit for the role.

---

## ✅ Features

- OpenAI-powered semantic evaluation (not just keyword matching)
- AI-generated summaries for recruiter-friendly results
- Upload and analyze multiple resumes simultaneously
- Clean and minimal UI built with Streamlit

---

## 🔐 Assumptions

- Resumes are typically **1–2 pages long**, which comfortably fits within the **context window of GPT-4o** (128k tokens).
- Therefore, the **text splitting step was removed** for simplicity and direct evaluation.
- Resumes are written in English and follow a general professional structure.
- OpenAI API key is stored securely via Streamlit Cloud's `secrets.toml` (not hardcoded).

---

## 📦 Setup (Local)

```bash
pip install -r requirements.txt
streamlit run main.py
