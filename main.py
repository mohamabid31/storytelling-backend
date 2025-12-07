import os
import re
import logging
from io import BytesIO
from typing import List, Optional

import boto3
import requests
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

from models import User, Base

# -------------------------------------------------------------------
# FastAPI app & static
# -------------------------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------------------------
# Logging & environment
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# OpenAI key for classic client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("🚨 ERROR: OPENAI_API_KEY is NOT set. Check environment variables!")
else:
    logger.info(f"✅ OPENAI_API_KEY detected (starts with): {api_key[:5]}…")
openai.api_key = api_key

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "YOUR_PIXABAY_API_KEY")

# -------------------------------------------------------------------
# DB setup (kept for compatibility)
# -------------------------------------------------------------------

DATABASE_URL = "postgresql://abid:yourstoryworld@localhost:5432/yourstoryworld"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -------------------------------------------------------------------
# CORS (includes production)
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# AWS Polly
# -------------------------------------------------------------------

polly = boto3.client("polly", region_name="eu-west-2")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def convert_to_ssml(word: str) -> str:
    """
    Convert a word into SSML for Polly with phoneme hints & syllable breaks.
    (Not wired in yet, but kept for future phonics TTS work.)
    """
    phonics_map = {
        "ai": "eɪ", "ay": "eɪ",
        "ee": "iː", "ea": "iː", "ie": "iː",
        "oa": "oʊ", "oe": "oʊ", "ue": "uː",
        "ow": "aʊ", "ou": "aʊ",
        "oi": "ɔɪ", "oy": "ɔɪ",
        "ar": "ɑː", "er": "ɜːr", "ir": "ɜːr", "or": "ɔːr", "ur": "ɜːr",
        "sh": "ʃ", "ch": "tʃ", "th": "θ", "wh": "w", "ph": "f",
        "bl": "bl", "cl": "kl", "fl": "fl", "gl": "ɡl", "pl": "pl", "sl": "sl",
        "br": "br", "cr": "kr", "dr": "dr", "fr": "fr", "gr": "ɡr", "pr": "pr", "tr": "tr",
        "sc": "sk", "sk": "sk", "sm": "sm", "sn": "sn", "sp": "sp", "st": "st", "sw": "sw",
        "qu": "kw",
    }

    for phonics, ipa in phonics_map.items():
        if phonics in word.lower():
            word = word.replace(
                phonics,
                f"<phoneme alphabet='ipa' ph='{ipa}'>{phonics}</phoneme>",
            )

    syllable_breaks = re.sub(
        r"([aeiouy]{1,2})(?![^<>]*>)",
        r"\1<break time='200ms'/>",
        word,
        flags=re.IGNORECASE,
    )
    return syllable_breaks


def split_text(text: str, max_length: int = 3000):
    """Split text into chunks for Polly."""
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


# -------------------------------------------------------------------
# Global config: word-limits & safety
# -------------------------------------------------------------------

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

SAFETY_BLOCK = (
    "The story must be fully child-safe and age-appropriate. Do NOT include any violence, "
    "gore, abuse, bullying, explicit or sexual content, nudity, self-harm, suicide, hate, "
    "discrimination, drug use, unsafe stunts, horror, or anything frightening or traumatic. "
    "No swearing or crude language. Keep the tone positive, kind, and encouraging."
)

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------

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


class ModerateRequest(BaseModel):
    text: str


# -------------------------------------------------------------------
# Abuse / safety filter endpoint (classic Moderation API)
# -------------------------------------------------------------------

@app.post("/moderate")
async def moderate_text(request: ModerateRequest):
    """
    Simple abuse / NSFW / safety filter using OpenAI Moderation.
    The frontend can call this explicitly if needed.
    """
    try:
        text = (request.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required for moderation.")

        logger.info("🔍 Running moderation on input (truncated): %s", text[:200])

        mod = openai.Moderation.create(
            model="text-moderation-latest",
            input=text,
        )

        result = mod["results"][0]
        return {
            "flagged": result["flagged"],
            "categories": result["categories"],
            "category_scores": result["category_scores"],
        }
    except Exception as e:
        logger.error("❌ Moderation error: %s", e)
        raise HTTPException(status_code=500, detail="Error running moderation.")


# -------------------------------------------------------------------
# Story generation (with safety + improved formatting)
# -------------------------------------------------------------------

@app.post("/generate")
async def generate_story(request: StoryRequest):
    try:
        logger.info(
            "✅ Story request — ageGroup=%s, length=%s, genre=%s",
            request.ageGroup,
            request.length,
            request.genre,
        )

        # Age-specific language constraints
        age_group_constraints = {
            "3-5 years old": (
                "Ensure every sentence is exactly 9 to 12 words long and only uses very simple words "
                "(about 4–5 letters). Avoid complex dialogue or long paragraphs. Keep each line short "
                "and clear while still making sense."
            ),
            "6-9 years old": (
                "Use simple words and short sentences. Introduce some new vocabulary but keep it easy to follow."
            ),
            "10-13 years old": "Use richer vocabulary with descriptive sentences and some dialogue.",
            "14+ years old": "Use more advanced vocabulary and more complex sentence structures.",
        }

        length_key = (request.length or "").lower()
        word_limit_instruction = CUSTOM_WORD_LIMITS.get(request.ageGroup, {}).get(length_key, "")

        story_description = (
            f"{request.storyDescription}\n"
            if request.storyDescription
            else "An adventure story.\n"
        )

        # Build base prompt
        prompt = (
            f"Write a {request.genre} story for a child in the age group {request.ageGroup}. "
            f"{story_description}"
            f"{age_group_constraints.get(request.ageGroup, '')} "
            f"{SAFETY_BLOCK} "
            f"{word_limit_instruction} "
            "\n\nFormatting rules:\n"
            "1. The very first line must be the title in this format:\n"
            "   Title: <Your Story Title>\n"
            "2. Then add a blank line.\n"
            "3. Then write the story as short paragraphs, separated by blank lines.\n"
            "4. At the very end, add a line like: [Word Count: X] where X is the actual number of words.\n"
        )

        if request.setting:
            prompt += f"\nThe story takes place in: {request.setting}."
        if request.themes:
            prompt += f"\nThemes to include: {', '.join(request.themes)}."
        if request.characterDetails:
            characters = ", ".join(
                [
                    f"{c.get('name', 'Unknown')} ({c.get('age', 'N/A')}): {c.get('description', '')}"
                    for c in request.characterDetails
                ]
            )
            prompt += f"\nCharacters: {characters}."

        if request.diversityMode:
            prompt += (
                "\nEnsure the story reflects diverse characters, cultures, and backgrounds in a natural, "
                "positive and respectful way."
            )

        # 🔍 Log prompt for debugging (truncated)
        logger.info("🧩 Story prompt (truncated): %s", prompt[:1000])

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # keep same as before; change if you like
            messages=[{"role": "user", "content": prompt}],
        )

        story_raw = response["choices"][0]["message"]["content"].strip()

        # --- Post-processing / formatting cleanup ---
        story = story_raw.replace("\r\n", "\n").strip()

        # Ensure title is on first line, and we capture it
        title = "Generated Story"
        title_match = re.search(r"^Title:\s*(.+)", story, flags=re.IGNORECASE | re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            first_line = story.split("\n", 1)[0].strip()
            if len(first_line) < 80:
                title = first_line

        logger.info("✅ Extracted title: %s", title)

        # Make sure there's a blank line after the title for consistent rendering
        lines = story.split("\n")
        if lines and lines[0].lower().startswith("title:") and (len(lines) == 1 or lines[1].strip() != ""):
            lines.insert(1, "")
            story = "\n".join(lines)

        return {"story": story, "title": title}

    except Exception as e:
        logger.error("❌ Error generating story: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Comprehension questions
# -------------------------------------------------------------------

@app.post("/generate_questions")
async def generate_questions(request: QuestionRequest):
    try:
        logger.info("✅ Question generation requested.")

        questions_prompt = f"""
Generate exactly 3 multiple-choice comprehension questions for the following story.
Each question should have 3 answer choices (A, B, C) and one correct answer.

Format each question as:
Question: <question text>
A) <answer choice>
B) <answer choice>
C) <answer choice>
Correct Answer: <correct option (A, B, or C)>

Story:
{request.story}
"""

        logger.info("🧩 Questions prompt (truncated): %s", questions_prompt[:800])

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": questions_prompt}],
        )

        raw_questions = response["choices"][0]["message"]["content"].strip()
        questions_list = raw_questions.split("\n\n")

        formatted_questions = []
        correct_answers = []

        for q in questions_list:
            if "Correct Answer:" in q:
                question_part, correct_answer = q.split("Correct Answer:", 1)
                correct_answer = correct_answer.strip().split()[0].replace(")", "")
                correct_answers.append(correct_answer)
            else:
                question_part = q
            formatted_questions.append(question_part.strip())

        logger.info("✅ Questions generated: %d", len(formatted_questions))
        logger.info("✅ Correct answers: %s", correct_answers)

        return {
            "comprehensionQuestions": formatted_questions,
            "correctAnswers": correct_answers,
        }

    except Exception as e:
        logger.error("❌ Error generating comprehension questions: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating comprehension questions: {str(e)}",
        )


# -------------------------------------------------------------------
# Image generation via Pixabay
# -------------------------------------------------------------------

@app.post("/generate_image")
async def generate_image(query: ImageRequest):
    try:
        search_query = (query.text or "").strip() or "fantasy illustration"

        logger.info("🎨 Fetching image for: %s", search_query)

        def pixabay_url(q: str) -> str:
            return (
                f"https://pixabay.com/api/?key={PIXABAY_API_KEY}"
                f"&q={q.replace(' ', '+')}"
                f"&image_type=illustration&editors_choice=true&safesearch=true"
            )

        response = requests.get(pixabay_url(search_query))
        if response.status_code != 200:
            raise Exception("Error fetching image from Pixabay API")

        data = response.json()

        if not data.get("hits"):
            logger.warning("⚠️ No images found for '%s'. Trying 'fantasy scene'.", search_query)
            response = requests.get(pixabay_url("fantasy scene"))
            data = response.json()

        image_url = data["hits"][0]["webformatURL"] if data.get("hits") else "/images/default.png"
        logger.info("✅ Image URL: %s", image_url)

        return {"image_url": image_url}

    except Exception as e:
        logger.error("❌ Error fetching image: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------------------------------------------------------
# TTS (full story)
# -------------------------------------------------------------------

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        text = (request.text or "").strip()
        if not text:
            raise ValueError("Text input is missing or empty.")

        logger.info("🔊 TTS request (truncated): %s", text[:120])

        text_chunks = split_text(text, 3000)
        audio_files = []

        for chunk in text_chunks:
            response = polly.synthesize_speech(
                Text=chunk,
                OutputFormat="mp3",
                VoiceId="Joanna",
            )
            if "AudioStream" in response:
                audio_files.append(response["AudioStream"].read())

        if not audio_files:
            raise Exception("No audio generated from Polly.")

        def generate_audio():
            for audio in audio_files:
                yield audio

        return StreamingResponse(generate_audio(), media_type="audio/mpeg")

    except Exception as e:
        logger.error("❌ TTS error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------------------------------------------------------
# Phonics transformation
# -------------------------------------------------------------------

@app.post("/generate_phonics")
async def generate_phonics(request: QuestionRequest):
    try:
        logger.info("✅ Phonics transformation requested.")

        phonics_prompt = f"""
Take the following story and:
- Break words into syllables using dashes (e.g., 'Ad-ven-ture').
- Wrap known phonics patterns ONLY in **double asterisks** (e.g., '**sh**ip', '**oa**k').
- Return only plain text, no HTML or explanations.

Story:
{request.story}
"""

        logger.info("🧩 Phonics prompt (truncated): %s", phonics_prompt[:800])

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": phonics_prompt}],
        )

        phonics_story = response["choices"][0]["message"]["content"].strip()
        logger.info("✅ Phonics story generated (truncated): %s", phonics_story[:200])

        return {"phonicsStory": phonics_story}

    except Exception as e:
        logger.error("❌ Error generating phonics story: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Phonics TTS (single word)
# -------------------------------------------------------------------

@app.post("/tts_phonics")
async def generate_phonics_tts(request: dict):
    text = (request.get("word") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing word for pronunciation.")

    try:
        syllables = re.sub(
            r"([aeiouy]+[^aeiouy]*)",
            r"\1<break time='250ms'/>",
            text,
            flags=re.IGNORECASE,
        )

        phonics_ssml = f"""
        <speak>
            <prosody rate="slow">
                {syllables}
            </prosody>
        </speak>
        """.strip()

        logger.info("🔍 Phonics SSML sent to Polly: %s", phonics_ssml)

        response = polly.synthesize_speech(
            Text=phonics_ssml,
            TextType="ssml",
            OutputFormat="mp3",
            VoiceId="Joanna",
        )

        audio_stream = response["AudioStream"].read()
        audio_buffer = BytesIO(audio_stream)

        return StreamingResponse(audio_buffer, media_type="audio/mpeg")

    except Exception as e:
        logger.error("❌ Polly phonics TTS error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------------------------------------------------------
# Local dev entrypoint
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
