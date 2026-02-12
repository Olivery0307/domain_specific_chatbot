import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 1. Setup CORS and Static Files (Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves the frontend at the root URL
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 2. Vertex AI Client Setup
# In Cloud Run, authentication is automatic via the service account.
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    logger.error("GOOGLE_CLOUD_PROJECT environment variable is not set")
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set")

LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

logger.info(f"Initializing Vertex AI client with project: {PROJECT_ID}, location: {LOCATION}")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# 3. Define the System Prompt (Project Requirement: Constraints & Escape Hatch)
SYSTEM_INSTRUCTION = """
You are an expert on [INSERT YOUR NICHE TOPIC HERE].
Your goal is to answer questions strictly related to this domain.

POSITIVE CONSTRAINTS:
- Only answer questions about [TOPIC].
- Provide answers in a concise, bulleted format.

ESCAPE HATCH:
- If the user asks about anything outside of [TOPIC], or if you are unsure, 
  respond with exactly: "I am sorry, but I can only answer questions about [TOPIC]."
"""

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "service": "domain-chatbot"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request: {request.message[:100]}...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3, # Low temp for deterministic factual answers
            ),
            contents=request.message
        )
        logger.info(f"Generated response successfully")
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal AI Error")

@app.get("/")
async def root():
    return {"message": "Chatbot API is running. Go to /static/index.html to chat."}
