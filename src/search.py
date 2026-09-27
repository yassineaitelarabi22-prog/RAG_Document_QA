import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================
# CONFIGURATION
# =========================

VECTORSTORE_PATH = "vectorstore"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 10


# =========================
# 1. Charger le modèle
# =========================

print("Chargement du modèle d'embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
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
# 3. Normalisation
# =========================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\sàâäéèêëîïôöùûüÿçœ-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================
# 4. Détection des mots-clés
# =========================

def extract_keywords(question):

    stopwords = {
        "quel",
        "quelle",
        "quels",
        "quelles",
        "est",
        "sont",
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "un",
        "une",
        "en",
        "à",
        "au",
        "aux",
        "et",
        "pour",
        "sur",
        "dans",
        "avec",
        "ce",
        "cette",
        "ces",
        "qui",
        "que",
        "combien",
        "comment",
        "année",
        "ans",
        "2023"
    }

    words = normalize_text(question).split()

    keywords = [
        word
        for word in words
        if word not in stopwords and len(word) > 2
    ]

    return keywords


# =========================
# 5. Score lexical
# =========================

def calculate_keyword_score(text, question, keywords):

    normalized_text = normalize_text(text)
    normalized_question = normalize_text(question)

    score = 0

    # ---------------------------------
    # Bonus pour les expressions exactes
    # ---------------------------------

    important_phrases = [
        "chiffres clés",
        "chiffre d affaires",
        "résultats financiers",
        "résultats financiers du groupe",
        "nombre de conteneurs",
        "tonnes de marchandises",
        "nombre d entreprises"
    ]

    for phrase in important_phrases:

        if phrase in normalized_question:

            if phrase in normalized_text:
                score += 10


    # ---------------------------------
    # Bonus mots-clés
    # ---------------------------------

    for keyword in keywords:

        if keyword in normalized_text:
            score += 1


    return score


# =========================
# 6. Question utilisateur
# =========================

question = input("\nPose ta question : ")

keywords = extract_keywords(question)

print("\nMots-clés détectés :", keywords)


# =========================
# 7. Recherche sémantique
# =========================

results = vectorstore.similarity_search_with_score(
    question,
    k=TOP_K
)


# =========================
# 8. Calcul du score final
# =========================

ranked_results = []

for document, semantic_score in results:

    text = document.page_content

    keyword_score = calculate_keyword_score(
        text,
        question,
        keywords
    )

    # Plus le score lexical est élevé,
    # plus le résultat est considéré comme pertinent.
    final_score = semantic_score - keyword_score

    ranked_results.append({
        "document": document,
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
        "final_score": final_score
    })


# =========================
# 9. Tri
# =========================

ranked_results.sort(
    key=lambda x: x["final_score"]
)


# =========================
# 10. Affichage
# =========================

print("\n")
print("=" * 60)
print("           PASSAGES PERTINENTS")
print("=" * 60)


for i, result in enumerate(ranked_results, start=1):

    document = result["document"]

    metadata = document.metadata

    print(f"\n--- Passage {i} ---")

    print(
        f"Page             : "
        f"{metadata.get('page', 'N/A')}"
    )

    print(
        f"Score sémantique : "
        f"{result['semantic_score']:.4f}"
    )

    print(
        f"Score mots-clés  : "
        f"{result['keyword_score']}"
    )

    print(
        f"Score final      : "
        f"{result['final_score']:.4f}"
    )

    print(
        f"Source           : "
        f"{metadata.get('source', 'N/A')}"
    )

    print("\n")

    print(document.page_content)

    print("\n" + "-" * 60)