from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Charger la base de connaissances
def charger_connaissances(fichier):
    with open(fichier, "r", encoding="utf-8") as f:
        contenu = f.read()
    # Découper en paragraphes
    paragraphes = [p.strip() for p in contenu.split("\n\n") if p.strip()]
    return paragraphes

# Créer la base vectorielle
def creer_base_vectorielle(paragraphes, modele):
    embeddings = modele.encode(paragraphes)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Trouver la meilleure réponse
def trouver_reponse(question, paragraphes, index, modele):
    question_vec = modele.encode([question]).astype("float32")
    _, indices = index.search(question_vec, k=2)
    reponses = [paragraphes[i] for i in indices[0]]
    return "\n\n".join(reponses)

# Programme principal
if __name__ == "__main__":
    print("⏳ Chargement du chatbot business...")
    
    # Charger le modèle multilingue (supporte le français)
    modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Charger les connaissances
    paragraphes = charger_connaissances("data/business_knowledge.txt")
    
    # Créer la base vectorielle
    index, _ = creer_base_vectorielle(paragraphes, modele)
    
    print("✅ Chatbot prêt ! Tape 'quitter' pour arrêter.\n")
    
    # Boucle de conversation
    while True:
        question = input("Toi : ")
        if question.lower() == "quitter":
            print("À bientôt ! 👋")
            break
        reponse = trouver_reponse(question, paragraphes, index, modele)
        print(f"\n🤖 Chatbot : {reponse}\n")