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
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_llm(temperature=0.2):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=temperature,
    )

def get_fast_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
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


_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriter for a data search engine.
Rewrite the user's question into a clear, specific search query.
- Fix spelling mistakes
- Resolve vague emotion words: "highly happy" → "show records with [sentiment: positive]"
- Resolve vague count words: "a few" → "5", "some" → "3"
- If the user asks for SOQL like SELECT ... FROM ..., convert it to plain English
- Keep it short — one sentence
- Return ONLY the rewritten query, nothing else"""),
    ("human", "{question}"),
])


def _rewrite_query(question: str) -> str:
    chain = _REWRITE_PROMPT | get_fast_llm() | StrOutputParser()
    return chain.invoke({"question": question})


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are Zarva, an intelligent data analyst assistant.
Answer questions using ONLY the data context below.
- Be specific: cite names, numbers, and values directly from the context
- If the user asks for one record, return only the single best match
- If the user asks for multiple, list them clearly
- If the answer is not in the context, say: "I couldn't find that in your data."
- Auto-correct intent: if the user seems to want something different from what they typed, answer what they meant
- Keep your tone professional but friendly
- If the user asks for standup updates or report summaries, format each record as: "HR-[ID] | Status: [value] | Owner: [value] | Update: [summary]"
- If the user asks to export or download, tell them to use the Export to Excel button below the chat

Conversation so far:
{history}

Data context:
{context}"""),
    ("human", "{question}"),
])


def ask(db: FAISS, question: str, history: list[dict] | None = None) -> str:
    rewritten = _rewrite_query(question)

    singular_words = {"one", "single", "top", "best", "highest", "most"}
    k = 1 if set(rewritten.lower().split()) & singular_words else 5
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(rewritten)
    context = "\n".join(d.page_content for d in docs)

    history_text = ""
    if history:
        for msg in history[-6:]:
            role = "User" if msg["role"] == "user" else "Zarva"
            history_text += f"{role}: {msg['content']}\n"

    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({
        "context": context,
        "question": rewritten,
        "history": history_text or "None",
    })
