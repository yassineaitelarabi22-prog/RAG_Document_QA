from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import ollama


# =========================
# Configuration
# =========================

VECTORSTORE_PATH = "vectorstore"
MODEL_NAME = "llama3.2:3b"


# =========================
# 1. Charger les embeddings
# =========================

print("Chargement du modèle d'embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================
# 2. Charger ChromaDB
# =========================

print("Chargement du vectorstore...")

vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)

print("Vectorstore chargé avec succès.")


# =========================
# 3. Fonction RAG
# =========================

def answer_question(question):

    # Recherche des passages pertinents
    results = vectorstore.similarity_search_with_score(
        question,
        k=5
    )

    if not results:
        return "Je ne trouve pas d'information pertinente dans le document."

    # Construire le contexte
    context_parts = []

    for i, (document, score) in enumerate(results, start=1):

        page = document.metadata.get("page", "?")
        source = document.metadata.get("source", "?")

        context_parts.append(
            f"""
--- DOCUMENT {i} ---
Source : {source}
Page : {page}

{document.page_content}
"""
        )

    context = "\n".join(context_parts)


    # =========================
    # 4. Prompt RAG
    # =========================

    prompt = f"""
Tu es un assistant spécialisé dans le question-réponse documentaire.

Tu dois répondre à la question uniquement à partir du contexte fourni
provenant du document.

Règles importantes :
- N'utilise pas tes connaissances générales.
- N'invente aucune information.
- Si la réponse n'est pas présente dans le contexte, dis :
  "Je ne trouve pas cette information dans le document."
- Réponds en français.
- Donne une réponse claire et concise.
- Lorsque c'est pertinent, indique la page du document utilisée.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
"""


    # =========================
    # 5. Appel à Ollama
    # =========================

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"], results


# =========================
# 6. Interface
# =========================

while True:

    question = input(
        "\nPose ta question (ou écris 'quit' pour quitter) : "
    )

    if question.lower() == "quit":
        print("Fin du programme.")
        break

    print("\nRecherche dans le document...")

    answer, results = answer_question(question)

    print("\n")
    print("=" * 60)
    print("                 RÉPONSE")
    print("=" * 60)

    print(answer)

    print("\n")
    print("=" * 60)
    print("                 SOURCES")
    print("=" * 60)

    # Afficher les sources utilisées
    displayed_sources = set()

    for document, score in results:

        page = document.metadata.get("page", "?")
        source = document.metadata.get("source", "?")

        source_key = (source, page)

        if source_key not in displayed_sources:

            print(f"- {source} | Page {page}")

            displayed_sources.add(source_key)