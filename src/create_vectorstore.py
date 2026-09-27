from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path("data/documents/rapport_tangermed_2023.pdf")
VECTORSTORE_PATH = "vectorstore"


# ============================================================
# 1. CHARGEMENT DU PDF
# ============================================================

print("Chargement du PDF...")

reader = PdfReader(PDF_PATH)

print(f"Nombre de pages : {len(reader.pages)}")


# ============================================================
# 2. EXTRACTION PAGE PAR PAGE
# ============================================================

documents = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    if text.strip():

        documents.append({
            "text": text,
            "page": page_number
        })


print(f"Pages contenant du texte : {len(documents)}")


# ============================================================
# 3. DÉCOUPAGE DU TEXTE
# ============================================================

print("Découpage du document...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=250,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


chunks = []
metadatas = []


for document in documents:

    page_number = document["page"]
    text = document["text"]

    page_chunks = text_splitter.split_text(text)

    for chunk in page_chunks:

        if not chunk.strip():
            continue

        chunks.append(chunk)

        metadatas.append({
            "source": "rapport_tangermed_2023.pdf",
            "page": page_number
        })


print(f"Nombre de chunks créés : {len(chunks)}")


# ============================================================
# 4. EMBEDDINGS
# ============================================================

print("Création des embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# 5. CRÉATION DU VECTORSTORE
# ============================================================

print("Création du vectorstore...")

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=VECTORSTORE_PATH
)


# ============================================================
# 6. FIN
# ============================================================

print()
print("============================================")
print("Vectorstore créé avec succès !")
print("============================================")
print(f"Chunks enregistrés : {len(chunks)}")
print(f"Emplacement        : {VECTORSTORE_PATH}")
print("============================================")