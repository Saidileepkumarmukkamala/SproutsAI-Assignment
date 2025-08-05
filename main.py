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


def build_qa_chain(documents):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7,api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa

def main(): 
    st.title("Candidate Recommendation Engine")
    job_description = st.text_area("Enter the Job Description")
    resume_files = st.file_uploader("Upload Candidate Resumes (PDF, DOCX, TXT)", accept_multiple_files=True)

    if st.button("Find Top Candidates"):
        if not job_description or not resume_files:
            st.error("Please provide both a job description and at least one resume.")
            return

        with st.spinner("Finding a match ........"):
            docs = load_documents(resume_files)
            chain = build_qa_chain(docs)

            results = []
            for file in resume_files:
                question = f"Is this resume a good fit for the following job description?\n{job_description}\nGive a concise summary about the fit. Be specific and concise."
                resume_docs = [d for d in docs if d.metadata["source"] == file.name]
                context = "\n".join([d.page_content for d in resume_docs])
                answer = chain.run({"query": question, "context": context})
                results.append({"Candidate": file.name, "LLM Recommendation": answer})

            df = pd.DataFrame(results)
            st.subheader("RAG Based Candidate Recommendation")
            for index, row in df.iterrows():
                with st.expander(row["Candidate"]):
                    st.markdown(f"**LLM Summary:** {row['LLM Recommendation']}")

if __name__ == '__main__':
    main()
