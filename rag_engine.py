import os
import pandas as pd
from dotenv import load_dotenv
from textblob import TextBlob
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
    df = pd.read_csv(file)
    docs = []
    for _, row in df.iterrows():
        line = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(line)
    return docs


def documents_from_text(file) -> list[str]:
    raw = file.read().decode("utf-8")
    chunk_size, overlap = 800, 100
    chunks = []
    start = 0
    while start < len(raw):
        chunks.append(raw[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def documents_from_salesforce(sf_connector, soql_queries: list[str]) -> list[str]:
    # TODO: wire up real SF connector
    docs = []
    for soql in soql_queries:
        for record in sf_connector.query(soql):
            line = " | ".join(f"{k}: {v}" for k, v in record.items() if k != "attributes")
            docs.append(line)
    return docs


def _sentiment_tag(text: str) -> str:
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"


def build_index(documents: list[str]) -> FAISS:
    tagged = [f"[sentiment: {_sentiment_tag(doc)}] {doc}" for doc in documents]
    return FAISS.from_texts(tagged, _embeddings)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a data analyst assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.
Be specific — cite numbers and names directly from the context.
If the user asks for one record, return only the single best match, not a list.

Context:
{context}"""),
    ("human", "{question}"),
])


_POSITIVE_WORDS = {"happy", "positive", "good", "great", "excellent", "joyful", "satisfied", "love", "amazing", "wonderful", "best", "fantastic"}
_NEGATIVE_WORDS = {"sad", "negative", "bad", "terrible", "awful", "angry", "unhappy", "hate", "worst", "horrible", "disappointed", "upset"}

def _enrich_query(question: str) -> str:
    words = set(question.lower().split())
    if words & _POSITIVE_WORDS:
        return question + " [sentiment: positive]"
    if words & _NEGATIVE_WORDS:
        return question + " [sentiment: negative]"
    return question


def ask(db: FAISS, question: str) -> str:
    singular_words = {"one", "single", "a", "any", "top", "best", "highest", "most"}
    k = 1 if set(question.lower().split()) & singular_words else 5
    retriever = db.as_retriever(search_kwargs={"k": k})
    enriched = _enrich_query(question)
    chain = (
        {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return chain.invoke(enriched)
