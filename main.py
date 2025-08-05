import streamlit as st
import os
import tempfile
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document
import pandas as pd
import shutil

st.set_page_config(page_title="Candidate Recommendation RAG", layout="wide")
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

@st.cache_resource
def load_llm():
    return ChatOpenAI(model_name="gpt-4o", temperature=0.7,api_key=openai_api_key)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(api_key=openai_api_key)

def save_uploaded_file(uploadedfile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploadedfile.read())
        return tmp.name

def load_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        temp_file_path = save_uploaded_file(file)
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(temp_file_path)
        else:
            continue
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = file.name
        documents.extend(docs)
    return documents

def build_vectorstore(documents):
    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb

def build_qa_chain(docs):
    vectordb1 = FAISS.from_documents(docs, get_embeddings())
    retriever = vectordb1.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = load_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa

def main():
    st.title("Candidate Recommendation Engine")
    job_description = st.text_area("Enter the Job Description", height=200)
    resume_files = st.file_uploader("Upload Candidate Resumes (PDF, DOCX, TXT)", accept_multiple_files=True)

    if st.button("Find Top Candidates"):
        if not job_description or not resume_files:
            st.error("Please provide both a job description and at least one resume.")
            return

        with st.spinner("Processing resumes and calculating similarity scores..."):
            all_docs = load_documents(resume_files)
            vectordb = build_vectorstore(all_docs)
            top_results = vectordb.similarity_search_with_score(job_description, k=5)

            top_resumes = []
            for doc, distance in top_results:
                similarity_score = round(1 - distance, 4)
                top_resumes.append({
                    "file_name": doc.metadata["source"],
                    "text": doc.page_content,
                    "score": similarity_score,
                    "doc": doc
                })

            qa_chain = build_qa_chain([r["doc"] for r in top_resumes])
            results = []

            for resume in top_resumes:
                question = f"Is this resume a good fit for the following job description?\n{job_description}\nGive a concise summary about the fit. Be specific and concise."
                response = qa_chain.invoke({"query": question, "context": resume["text"]})
                answer = response["result"] if "result" in response else "No response generated."

                results.append({
                    "Candidate": resume["file_name"],
                    "Similarity Score": resume["score"],
                    "LLM Recommendation": answer
                })

            df = pd.DataFrame(results).sort_values(by="Similarity Score", ascending=False)
            st.subheader("Top 5 Candidates")
            st.dataframe(df[["Candidate", "Similarity Score"]])

            for index, row in df.iterrows():
                with st.expander(row["Candidate"]):
                    st.markdown(f"**Similarity Score:** {row['Similarity Score']}")
                    st.markdown(f"**LLM Summary:** {row['LLM Recommendation']}")

if __name__ == '__main__':
    main()
