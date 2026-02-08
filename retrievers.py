import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import WikipediaRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# ----------------------------
# ENV CHECK
# ----------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# ----------------------------
# 1. Wikipedia Retriever
# ----------------------------
wiki_retriever = WikipediaRetriever(top_k_results=2, lang="en")
wiki_docs = wiki_retriever.invoke(
    "Geopolitical history of India and Pakistan from a Chinese perspective"
)

print("\n=== Wikipedia Results ===")
for i, doc in enumerate(wiki_docs, 1):
    print(f"\n[{i}] {doc.page_content[:300]}...")


# ----------------------------
# 2. FAISS Similarity Retriever
# ----------------------------
docs = [
    Document(page_content="LangChain helps developers build LLM applications."),
    Document(page_content="Chroma is a vector database optimized for embeddings."),
    Document(page_content="FAISS enables fast similarity search."),
    Document(page_content="Embeddings convert text into vectors."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

similarity_retriever = vectorstore.as_retriever(k=2)

print("\n=== Similarity Search ===")
for doc in similarity_retriever.invoke("What is FAISS?"):
    print("-", doc.page_content)


# ----------------------------
# 3. MMR Retriever
# ----------------------------
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5},
)

print("\n=== MMR Results ===")
for doc in mmr_retriever.invoke("Explain LangChain"):
    print("-", doc.page_content)


# ----------------------------
# 4. Multi Query Retriever
# ----------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(k=3),
    llm=llm
)

print("\n=== Multi Query Results ===")
for doc in multi_query_retriever.invoke("How do embeddings work?"):
    print("-", doc.page_content)


# ----------------------------
# 5. Contextual Compression Retriever
# ----------------------------
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=vectorstore.as_retriever(k=5),
    base_compressor=compressor,
)

print("\n=== Compressed Results ===")
for doc in compression_retriever.invoke("What is LangChain?"):
    print("-", doc.page_content)
