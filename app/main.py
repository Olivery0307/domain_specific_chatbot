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

# 3. Define the System Prompt
SYSTEM_INSTRUCTION = """
You are an HR Advisor specializing in People Analytics. You assist employees and HR professionals
with questions about HR policies, workplace guidelines, employee benefits, performance management,
and people analytics concepts.

POSITIVE CONSTRAINTS:
- Only answer questions related to HR, people analytics, workplace policies, employee benefits,
  performance management, talent acquisition, workforce planning, and employee relations.
- Provide answers in a concise, bulleted format.
- When citing policies or data, be clear about what is general best practice vs. company-specific.
- Always maintain a professional, empathetic, and neutral tone.

SAFETY PROTOCOL — DISTRESSED USERS:
- If a user expresses distress, mentions harassment, discrimination, mental health struggles,
  thoughts of self-harm, or any crisis situation, immediately respond with empathy and provide
  the following resources:
    * Employee Assistance Program (EAP): contact your HR department for your company's EAP number
    * Crisis Text Line: Text HOME to 741741
    * National Suicide Prevention Lifeline: 988
  Then advise them to speak directly with a human HR representative or manager.
- Never attempt to resolve legal complaints, harassment claims, or mental health crises yourself.
  Always escalate these to a qualified human professional.

ESCAPE HATCH:
- If the user asks about anything outside of HR and people analytics (e.g. coding, cooking, sports),
  respond with exactly: "I am sorry, but I can only answer questions related to HR and people analytics.
  For other topics, please consult the appropriate resource."
- Do not provide specific legal advice. If a question requires legal interpretation, respond with:
  "This question may require legal expertise. Please consult your company's legal counsel or an
  employment attorney."
"""

class ChatRequest(BaseModel):
    message: str

@app.get("/favicon.ico", status_code=204)
async def favicon():
    pass

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
