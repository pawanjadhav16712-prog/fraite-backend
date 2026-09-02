import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from duckduckgo_search import DDGS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    prompt: str

@app.post("/api/chat")
async def chat(query: Query):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Gemini API Key missing")
        
        client = genai.Client(api_key=api_key)
        
        # Web Search via DuckDuckGo
        with DDGS() as ddgs:
            results = list(ddgs.text(query.prompt, max_results=3))
            search_context = "\n".join([f"- {r['title']}: {r['body']}" for r in results]) if results else "No web results."
        
        full_prompt = f"User Request: {query.prompt}\n\nWeb Search Context:\n{search_context}"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
