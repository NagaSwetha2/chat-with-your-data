import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_llm(temperature=0.2):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=temperature,
    )


def documents_from_csv(file) -> list[str]:
    """Each row becomes one searchable document. Works for any CSV shape."""
    df = pd.read_csv(file)
    docs = []
    for _, row in df.iterrows():
        line = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(line)
    return docs


def documents_from_text(file) -> list[str]:
    """Chunk plain text into overlapping windows for retrieval."""
    raw = file.read().decode("utf-8")
    chunk_size, overlap = 800, 100
    chunks = []
    start = 0
    while start < len(raw):
        chunks.append(raw[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def documents_from_salesforce(sf_connector, soql_queries: list[str]) -> list[str]:
    """
    Placeholder for the Phase 2 Salesforce connector.
    sf_connector.query(soql) should return a list of record dicts.
    Each record becomes one searchable document, same shape as CSV rows.
    """
    docs = []
    for soql in soql_queries:
        for record in sf_connector.query(soql):
            line = " | ".join(f"{k}: {v}" for k, v in record.items() if k != "attributes")
            docs.append(line)
    return docs


def build_index(documents: list[str]) -> FAISS:
    return FAISS.from_texts(documents, _embeddings)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a data analyst assistant. Answer the question using
ONLY the context below. If the answer isn't in the context, say you don't know.
Be specific — cite numbers and names directly from the context.

Context:
{context}"""),
    ("human", "{question}"),
])


def ask(db: FAISS, question: str) -> str:
    retriever = db.as_retriever(search_kwargs={"k": 5})
    chain = (
        {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return chain.invoke(question)
