import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import ollama


# =========================
# Configuration
# =========================

VECTORSTORE_PATH = "vectorstore"
MODEL_NAME = "llama3.2:3b"


# =========================
# Interface
# =========================

st.set_page_config(
    page_title="Assistant documentaire",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Assistant documentaire RAG")
st.write(
    "Posez une question sur le rapport annuel 2023 de Tanger Med."
)


# =========================
# Chargement des embeddings
# =========================

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )

    return vectorstore


vectorstore = load_vectorstore()


# =========================
# Question
# =========================

question = st.text_input(
    "Posez votre question :",
    placeholder="Exemple : Combien d'entreprises sont installées dans les zones d'activité ?"
)


if question:

    with st.spinner("Recherche dans le document..."):

        # Recherche
        results = vectorstore.similarity_search_with_score(
            question,
            k=5
        )

        # Construction du contexte
        context_parts = []

        for i, (document, score) in enumerate(results, start=1):

            page = document.metadata.get("page", "?")
            source = document.metadata.get("source", "?")

            context_parts.append(
                f"""
DOCUMENT {i}
Source : {source}
Page : {page}

{document.page_content}
"""
            )

        context = "\n".join(context_parts)


        # Prompt
        prompt = f"""
Tu es un assistant spécialisé dans le question-réponse documentaire.

Réponds uniquement à partir du contexte fourni.

Règles :
- N'utilise pas tes connaissances générales.
- N'invente aucune information.
- Si l'information n'est pas présente, dis :
  "Je ne trouve pas cette information dans le document."
- Réponds en français.
- Réponds clairement et directement.
- Indique la page utilisée lorsque c'est pertinent.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
"""


        # Ollama
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]


    # =========================
    # Affichage réponse
    # =========================

    st.subheader("💬 Réponse")

    st.write(answer)


    # =========================
    # Sources
    # =========================

    st.subheader("📖 Sources")

    displayed_sources = set()

    for document, score in results:

        page = document.metadata.get("page", "?")
        source = document.metadata.get("source", "?")

        source_key = (source, page)

        if source_key not in displayed_sources:

            st.write(
                f"📄 {source} — Page {page}"
            )

            displayed_sources.add(source_key)