from pathlib import Path
from pypdf import PdfReader


# =========================
# Configuration
# =========================

PDF_PATH = Path("data/documents/rapport_tangermed_2023.pdf")


# =========================
# 1. Charger le PDF
# =========================

reader = PdfReader(PDF_PATH)

print(f"Nombre de pages : {len(reader.pages)}")


# =========================
# 2. Extraire le texte
# =========================

text = ""

for i, page in enumerate(reader.pages, start=1):

    page_text = page.extract_text() or ""

    text += page_text + "\n"


print(f"Nombre de caractères extraits : {len(text)}")


# =========================
# 3. Afficher un aperçu
# =========================

print("\n===== APERÇU DU DOCUMENT =====\n")

print(text[:3000])


# =========================
# 4. Rechercher "NOS CHIFFRES CLÉS"
# =========================

search_term = "NOS CHIFFRES CLÉS"

position = text.upper().find(search_term)

if position != -1:

    print("\n==============================================")
    print("TEXTE AUTOUR DE 'NOS CHIFFRES CLÉS'")
    print("==============================================\n")

    print(text[position:position + 5000])

else:

    print("\n'NOS CHIFFRES CLÉS' non trouvé dans le texte extrait.")