import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

# Charger la clé API
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---- Configuration de la page ----
st.set_page_config(
    page_title="BizBot — Assistant Business",
    page_icon="💼",
    layout="centered"
)

# ---- CSS Personnalisé ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=DM+Sans:wght@300;400;500&display=swap');

/* Fond général */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #111827 50%, #0d1117 100%);
    font-family: 'DM Sans', sans-serif;
}

/* Cacher éléments Streamlit par défaut */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Titre principal */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(90deg, #f59e0b, #fcd34d, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.main-header p {
    color: #6b7280;
    font-size: 0.95rem;
    font-weight: 300;
    margin-top: 0.3rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Séparateur */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #f59e0b44, transparent);
    margin: 1rem 0 2rem 0;
}

/* Messages utilisateur */
.user-bubble {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: white;
    padding: 1rem 1.4rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3);
}

/* Messages chatbot */
.bot-bubble {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #f59e0b33;
    color: #e5e7eb;
    padding: 1rem 1.4rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.7;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

/* Labels */
.label-user {
    text-align: right;
    color: #3b82f6;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.5rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.label-bot {
    color: #f59e0b;
    font-size: 0.75rem;
    font-weight: 500;
    margin-left: 0.5rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Zone de saisie */
.stChatInputContainer {
    background: #111827 !important;
    border: 1px solid #f59e0b44 !important;
    border-radius: 16px !important;
}
.stChatInput textarea {
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Stats bar */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2rem;
    padding: 1rem;
    background: #ffffff08;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.stat-item {
    text-align: center;
}
.stat-number {
    font-family: 'Playfair Display', serif;
    color: #f59e0b;
    font-size: 1.4rem;
}
.stat-label {
    color: #6b7280;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ---- Chargement modèle ----
@st.cache_resource
def charger_modele():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource
def charger_connaissances():
    with open(
        "data/business_knowledge.txt", "r", encoding="utf-8"
    ) as f:
        contenu = f.read()
    paragraphes = [
        p.strip()
        for p in contenu.split("\n\n")
        if p.strip()
    ]
    return paragraphes

def creer_index(paragraphes, modele):
    embeddings = modele.encode(paragraphes)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def trouver_contexte(question, paragraphes, index, modele):
    question_vec = modele.encode([question]).astype("float32")
    _, indices = index.search(question_vec, k=2)
    return "\n\n".join([paragraphes[i] for i in indices[0]])

def generer_reponse(question, contexte):
    prompt = f"""Tu es BizBot, un expert en business et entrepreneuriat.
Utilise ce contexte pour répondre en français de façon claire,
structurée et professionnelle. Sois concis mais complet.

Contexte :
{contexte}

Question : {question}

Réponds de façon naturelle et professionnelle en français."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

# ---- En-tête ----
st.markdown("""
<div class="main-header">
    <h1>💼 BizBot</h1>
    <p>Votre assistant intelligent en business & entrepreneuriat</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ---- Chargement ----
modele = charger_modele()
paragraphes = charger_connaissances()
index = creer_index(paragraphes, modele)

# ---- Stats ----
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-number">{len(paragraphes)}</div>
        <div class="stat-label">Sujets maîtrisés</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">24/7</div>
        <div class="stat-label">Disponibilité</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">🇫🇷</div>
        <div class="stat-label">En français</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Historique ----
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Bonjour ! Je suis **BizBot**, votre expert en business & entrepreneuriat. Posez-moi n'importe quelle question sur la création d'entreprise, le marketing, la finance ou la stratégie !"
    }]

# ---- Afficher messages ----
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="label-user">Vous</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="label-bot">💼 BizBot</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-bubble">{message["content"]}</div>', unsafe_allow_html=True)

# ---- Saisie ----
question = st.chat_input("Posez votre question business...")

if question:
    st.session_state.messages.append({
        "role": "user", "content": question
    })
    st.markdown(f'<div class="label-user">Vous</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="user-bubble">{question}</div>', unsafe_allow_html=True)

    with st.spinner("BizBot réfléchit... 🧠"):
        contexte = trouver_contexte(
            question, paragraphes, index, modele
        )
        reponse = generer_reponse(question, contexte)

    st.session_state.messages.append({
        "role": "assistant", "content": reponse
    })
    st.markdown(f'<div class="label-bot">💼 BizBot</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-bubble">{reponse}</div>', unsafe_allow_html=True)
    st.rerun()