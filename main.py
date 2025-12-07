import openai
import os
import requests
import boto3
import logging
import re
from io import BytesIO

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from typing import List, Optional

# ------------------------------------------------------------------
# FASTAPI INITIALISATION
# ------------------------------------------------------------------

app = FastAPI()

# Serve static files (MP3 output)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load environment
load_dotenv()

# ------------------------------------------------------------------
# CORS – MUST INCLUDE YOUR PRODUCTION DOMAIN
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://yourstoryworld.com",
        "https://www.yourstoryworld.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# OpenAI Setup
# ------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    logger.error("🚨 OPENAI_API_KEY missing.")
else:
    logger.info(f"🔑 OpenAI key loaded ({api_key[:5]}*******)")

openai.api_key = api_key

# ------------------------------------------------------------------
# AWS POLLY
# ------------------------------------------------------------------

polly = boto3.client("polly", region_name="eu-west-2")

# ------------------------------------------------------------------
# Pixabay
# ------------------------------------------------------------------

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")

# ------------------------------------------------------------------
# Database (Unused now but kept for compatibility)
# ------------------------------------------------------------------

DATABASE_URL = "postgresql://abid:yourstoryworld@localhost:5432/yourstoryworld"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# ------------------------------------------------------------------
# GLOBAL CONSTANTS
# ------------------------------------------------------------------

CUSTOM_WORD_LIMITS = {
    "3-5 years old": {
        "short": "Make sure the story is exactly 50 words.",
        "medium": "Keep the story between 150 and 250 words.",
        "long": "Keep the story between 400 and 600 words.",
    },
    "6-9 years old": {
        "short": "Make sure the story is around 150 words.",
        "medium": "Keep the story between 400 and 600 words.",
        "long": "Keep the story between 700 and 850 words.",
    },
    "10-13 years old": {
        "short": "Make sure the story is around 400 words.",
        "medium": "Keep the story between 700 and 850 words.",
        "long": "Keep the story between 950 and 1100 words.",
    },
    "14+ years old": {
        "short": "Make sure the story is around 500 words.",
        "medium": "Keep the story between 800 and 950 words.",
        "long": "Keep the story between 1200 and 1400 words.",
    },
}

# ------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------

class StoryRequest(BaseModel):
    genre: str
    setting: Optional[str] = None
    themes: List[str] = []
    length: str
    ageGroup: str
    storyDescription: Optional[str] = None
    characterDetails: List[dict] = []
    phonicsAligned: Optional[bool] = False
    diversityMode: Optional[bool] = False


class QuestionRequest(BaseModel):
    story: str


class ImageRequest(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def split_text(text, max_length=3000):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


# ------------------------------------------------------------------
# STORY GENERATION ENDPOINT
# ------------------------------------------------------------------

@app.post("/generate")
async def generate_story(request: StoryRequest):
    try:
        logger.info("📘 Story request received")

        age_rules = {
            "3-5 years old": (
                "Ensure every sentence is exactly 9–12 words long using only simple words "
                "(4–5 letters max). Avoid complex dialogue."
            ),
            "6-9 years old": "Use simple, clear sentences with some new vocabulary.",
            "10-13 years old": "Use richer vocabulary and descriptive language.",
            "14+ years old": "Use advanced vocabulary and complex structures.",
        }

        # Word-limit selection
        length_key = (request.length or "").lower()
        word_limit_instruction = CUSTOM_WORD_LIMITS.get(request.ageGroup, {}).get(length_key, "")

        # Add story description if provided
        description_text = (
            f"{request.storyDescription}\n" if request.storyDescription else "An adventure story.\n"
        )

        # Build prompt
        prompt = (
            f"Write a {request.genre} story for {request.ageGroup}. "
            f"{description_text}"
            f"{age_rules.get(request.ageGroup, '')} "
            f"{word_limit_instruction} "
            "\n\nStart the story with a title like:\n"
            "Title: <Your Story Title>\n\n"
            "End the story with '[Word Count: X]'."
        )

        if request.setting:
            prompt += f" The story takes place in {request.setting}."

        if request.themes:
            prompt += f" Themes include: {', '.join(request.themes)}."

        if request.characterDetails:
            characters = ", ".join(
                [f"{c['name']} ({c['age']}): {c['description']}" for c in request.characterDetails]
            )
            prompt += f" Characters: {characters}."

        # OpenAI call
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        story = response["choices"][0]["message"]["content"]

        # Extract title
        title = "Generated Story"
        match = re.search(r"Title:\s*(.+)", story)
        if match:
            title = match.group(1).strip()
        else:
            first_line = story.split("\n")[0].strip()
            if len(first_line) < 60:
                title = first_line

        return {"story": story, "title": title}

    except Exception as e:
        logger.error(f"❌ Story error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# COMPREHENSION QUESTIONS
# ------------------------------------------------------------------

@app.post("/generate_questions")
async def generate_questions(request: QuestionRequest):
    try:
        prompt = f"""
Generate 3 multiple-choice comprehension questions for this story.
Each question MUST have options A, B, C and specify the correct one.

Story:
{request.story}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response["choices"][0]["message"]["content"].strip()
        blocks = raw.split("\n\n")

        questions = []
        answers = []

        for block in blocks:
            if "Correct Answer:" in block:
                q_text, ans = block.split("Correct Answer:")
                ans = ans.strip()[0]
                answers.append(ans)
                questions.append(q_text.strip())

        return {
            "comprehensionQuestions": questions,
            "correctAnswers": answers,
        }

    except Exception as e:
        logger.error(f"❌ Question generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# PIXABAY IMAGE FETCH
# ------------------------------------------------------------------

@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    try:
        query = request.text.strip() or "fantasy illustration"

        url = (
            f"https://pixabay.com/api/?key={PIXABAY_API_KEY}"
            f"&q={query.replace(' ', '+')}"
            f"&image_type=illustration&editors_choice=true&safesearch=true"
        )

        r = requests.get(url)
        data = r.json()

        if not data.get("hits"):
            return {"image_url": "/images/default.png"}

        return {"image_url": data["hits"][0]["webformatURL"]}

    except Exception as e:
        logger.error(f"❌ Image error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------------
# TEXT TO SPEECH (FULL STORY)
# ------------------------------------------------------------------

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        text = request.text.strip()
        if not text:
            raise ValueError("Missing text")

        chunks = split_text(text)
        audio_parts = []

        for chunk in chunks:
            r = polly.synthesize_speech(Text=chunk, OutputFormat="mp3", VoiceId="Joanna")
            audio_parts.append(r["AudioStream"].read())

        def stream():
            for part in audio_parts:
                yield part

        return StreamingResponse(stream(), media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------------
# PHONICS STORY PROCESSING
# ------------------------------------------------------------------

@app.post("/generate_phonics")
async def generate_phonics(request: QuestionRequest):
    try:
        prompt = f"""
Rewrite this story with:
- Words split into syllables using dashes (Ad-ven-ture)
- Known phonics patterns wrapped in **double asterisks**
Return ONLY the modified story.

Story:
{request.story}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        phonics = response["choices"][0]["message"]["content"].strip()
        return {"phonicsStory": phonics}

    except Exception as e:
        logger.error(f"❌ Phonics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# PHONICS WORD-BY-WORD TTS
# ------------------------------------------------------------------

@app.post("/tts_phonics")
async def generate_phonics_tts(request: dict):
    word = request.get("word", "").strip()
    if not word:
        raise HTTPException(400, "Missing word")

    try:
        syllables = re.sub(
            r"([aeiouy]+[^aeiouy]*)",
            r"\1<break time='250ms'/>",
            word,
            flags=re.IGNORECASE
        )

        ssml = f"""
<speak>
    <prosody rate="slow">
        {syllables}
    </prosody>
</speak>
"""

        r = polly.synthesize_speech(
            Text=ssml,
            TextType="ssml",
            OutputFormat="mp3",
            VoiceId="Joanna"
        )

        return StreamingResponse(BytesIO(r["AudioStream"].read()), media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"❌ Phonics TTS error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------------
# RUN SERVER (DEV ONLY)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
