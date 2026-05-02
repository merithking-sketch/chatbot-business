from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import json

PORT = int(os.environ.get("PORT", 8000))

load_dotenv()
app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("⏳ Chargement du modèle...")
modele = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/business_knowledge.txt", "r", encoding="utf-8") as f:
    contenu = f.read()

paragraphes = [p.strip() for p in contenu.split("\n\n") if p.strip()]
embeddings = modele.encode(paragraphes).astype("float32")
index_faiss = faiss.IndexFlatL2(embeddings.shape[1])
index_faiss.add(embeddings)
print(f"✅ Prêt ! {len(paragraphes)} sujets chargés.")

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    q_vec = modele.encode([request.message]).astype("float32")
    _, indices = index_faiss.search(q_vec, k=2)
    contexte = "\n\n".join([paragraphes[i] for i in indices[0]])

    messages = [{
        "role": "system",
        "content": f"""Tu es BizBot, expert en business et
entrepreneuriat francophone. Utilise ce contexte si pertinent:
{contexte}. Réponds en français, de façon claire et professionnelle."""
    }]

    for msg in request.history[-10:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    messages.append({"role": "user", "content": request.message})

    def stream_response():
        try:
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=600,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'text': f'Erreur: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )