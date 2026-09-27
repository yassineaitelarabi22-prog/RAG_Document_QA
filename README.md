# Assistant documentaire RAG – Tanger Med

## Description

Projet d'assistant documentaire basé sur l'architecture RAG
(Retrieval-Augmented Generation).

L'application permet à un utilisateur de poser des questions
sur le rapport annuel 2023 de Tanger Med et d'obtenir une
réponse générée à partir des passages pertinents du document.

## Fonctionnalités

- Recherche sémantique dans un document PDF
- Extraction des passages pertinents
- Utilisation d'embeddings
- Recherche dans un vectorstore
- Génération de réponses avec un LLM local
- Utilisation d'Ollama et du modèle Llama 3.2
- Affichage des sources et des pages
- Interface web pour poser les questions

## Architecture

PDF → Extraction → Chunking → Embeddings → Vectorstore
→ Recherche sémantique → Contexte → LLM → Réponse

## Technologies

- Python
- LangChain
- Hugging Face
- Embeddings
- Vectorstore
- Ollama
- Llama 3.2
- RAG
- Streamlit / interface web

## Exemple

Question :

"Combien d'entreprises sont présentes dans les zones
d'activités de Tanger Med ?"

Réponse :

"Selon le document, il y a 1 300 entreprises installées
dans les zones d'activité de Tanger Med."

Source : Rapport annuel 2023, page 52.

## Installation

Créer un environnement virtuel :

```bash
python -m venv .venv