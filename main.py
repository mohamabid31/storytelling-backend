import openai
import os
import requests
import boto3
import logging
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from typing import List, Optional

from models import User, Base
from dotenv import load_dotenv

# ✅ Initialize FastAPI app
app = FastAPI()  # ✅ Define the app instance first

# Serve static files (MP3s)
app.mount("/static", StaticFiles(directory="static"), name="static")

import re

import re

def convert_to_ssml(word):
    """
    Convert a word into SSML for Polly with:
    - Natural pronunciation
    - Syllable breaks
    - Correct phoneme replacements without breaking SSML
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
        "qu": "kw"
    }

    # ✅ Step 1: Replace phonemes correctly
    for phonics, ipa in phonics_map.items():
        if phonics in word.lower():
            word = word.replace(
                phonics,
                f"<phoneme alphabet='ipa' ph='{ipa}'>{phonics}</phoneme>"
            )

    # ✅ Step 2: Insert **syllable breaks** without affecting phonemes
    # - Only apply breaks between vowels, **EXCEPT inside phoneme tags**.
    syllable_breaks = re.sub(
        r"([aeiouy]{1,2})(?![^<>]*>)",  # Avoid replacing inside <phoneme> tags
        r"\1<break time='200ms'/>",
        word,
        flags=re.IGNORECASE
    )

    return syllable_breaks

# ✅ Initialize Polly with a specific region
polly = boto3.client("polly", region_name="eu-west-2")  # Ensure your AWS region is correct

# ✅ Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("🚨 ERROR: OPENAI_API_KEY is NOT set. Check environment variables!")
else:
    logger.info(f"✅ API Key detected: {api_key[:5]}...********")
openai.api_key = api_key

# ✅ Pixabay API Key
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "YOUR_PIXABAY_API_KEY")

# ✅ Database Setup
DATABASE_URL = "postgresql://abid:yourstoryworld@localhost:5432/yourstoryworld"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # ✅ Allow local frontend
        "https://yourstoryworld-imqyg0cm6-mohammed-abids-projects.vercel.app",  # ✅ Allow deployed frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # ✅ Allow all headers
)


# ✅ Pydantic Schemas
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

# ✅ Function to split text for Polly (if longer than 3000 characters)
def split_text(text, max_length=3000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# ✅ Story Generation Route
@app.post("/generate")
async def generate_story(request: StoryRequest):
    try:
        logger.info("✅ Received a story generation request")

        # ✅ Define language rules per age group
        age_group_constraints = {
            "3-5 years old": "Use only simple words (max 4-5 letters). Short sentences (5-8 words).",
            "6-9 years old": "Use simple words, but introduce some new words. Sentences should be easy to read.",
            "10-13 years old": "Use richer vocabulary with descriptive sentences.",
            "14+ years old": "Use advanced vocabulary and complex sentence structures.",
        }

        # ✅ Adjust word limit
        word_limits = {
            "short": "Make sure the story is **exactly 50 words**.",
            "medium": "Keep the story between **150 and 250 words**.",
            "long": "Keep the story between **400 and 600 words**."
        }

        # ✅ Construct prompt
        prompt = (
            f"Write a {request.length}-word {request.genre} story for {request.ageGroup}. "
            f"The story should be about {request.storyDescription or 'an adventure'}. "
            f"{age_group_constraints.get(request.ageGroup, '')} "
            f"{word_limits.get(request.length.lower(), 'Ensure the story is well-structured and engaging.')}"
            f"\n\nAt the start of the story, include a clear title on the first line, like this:"
            f"\nTitle: <Your Story Title>\n\n"
            f"At the end of the story, add '[Word Count: X]' where X is the actual number of words."
        )

        if request.setting:
            prompt += f" The story takes place in {request.setting}."
        if request.themes:
            prompt += f" Themes include: {', '.join(request.themes)}."
        if request.characterDetails:
            characters = ", ".join([f"{c['name']} ({c['age']}): {c['description']}" for c in request.characterDetails])
            prompt += f" Characters: {characters}."

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": prompt}]
)



        story = response["choices"][0]["message"]["content"]

        # ✅ Extract Title from AI Response
        title = "Generated Story"
        title_match = re.search(r"Title:\s*(.+)", story)

        if title_match:
            title = title_match.group(1).strip()
        else:
            # If the title isn't explicitly labeled, assume the first line is the title
            first_line = story.split("\n")[0].strip()
            if len(first_line) < 60 and " " in first_line:  # Make sure it's a reasonable title
                title = first_line

        logger.info(f"✅ Extracted Story Title: {title}")

        return {"story": story, "title": title}

    except Exception as e:
        logger.error(f"❌ Error generating story: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_questions")
async def generate_questions(request: QuestionRequest):
    try:
        logger.info("✅ Received a question generation request")

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

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": questions_prompt}]
        )

        if "choices" in response and len(response["choices"]) > 0:
            raw_questions = response["choices"][0]["message"]["content"].strip()
            questions_list = raw_questions.split("\n\n")  # Splitting each question block

            formatted_questions = []
            correct_answers = []

            for q in questions_list:
                if "Correct Answer:" in q:
                    question_part, correct_answer = q.split("Correct Answer:")
                    correct_answer = correct_answer.strip().split()[0].replace(")", "")  # ✅ Fix: Remove ) if present
                    correct_answers.append(correct_answer)
                else:
                    question_part = q

                formatted_questions.append(question_part.strip())

            logger.info(f"✅ Generated Questions: {formatted_questions}")
            logger.info(f"✅ Correct Answers Extracted: {correct_answers}")  # 🔴 Log correct answers

        else:
            logger.warning("⚠️ No comprehension questions returned by OpenAI.")
            formatted_questions = []
            correct_answers = []

        return {
            "comprehensionQuestions": formatted_questions,
            "correctAnswers": correct_answers  # ✅ Now only stores "A", "B", or "C"
        }
    except Exception as e:
        logger.error(f"❌ Error generating comprehension questions: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating comprehension questions: {str(e)}")


# ✅ Image Generation Route
@app.post("/generate_image")
async def generate_image(query: ImageRequest):
    try:
        search_query = query.text.strip()
        if not search_query:
            search_query = "fantasy illustration"  # Provide a default query

        logger.info(f"🔍 Fetching image for: {search_query}")
        # ✅ Make API request to Pixabay
        pixabay_url = (
            f"https://pixabay.com/api/?key={PIXABAY_API_KEY}"
            f"&q={search_query.replace(' ', '+')}"
            f"&image_type=illustration&editors_choice=true&safesearch=true"
        )
        response = requests.get(pixabay_url)

        if response.status_code != 200:
            raise Exception("Error fetching image from Pixabay API")

        data = response.json()

        # ✅ If no images found, try a fallback search
        if not data.get("hits"):
            logger.warning(f"⚠️ No images found for {search_query}. Trying 'fantasy scene'.")
            fallback_url = (
                f"https://pixabay.com/api/?key={PIXABAY_API_KEY}"
                f"&q=fantasy+scene&image_type=illustration&editors_choice=true&safesearch=true"
            )
            response = requests.get(fallback_url)
            data = response.json()

        image_url = data["hits"][0]["webformatURL"] if data.get("hits") else "/images/default.png"
        logger.info(f"✅ Image URL: {image_url}")

        return {"image_url": image_url}

    except Exception as e:
        logger.error(f"❌ Error fetching image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        if not request.text.strip():
            raise ValueError("Text input is missing or empty.")

        logger.info(f"🔊 Processing TTS for text: {request.text[:100]}...")  # Debug log

        text_chunks = split_text(request.text, 3000)
        audio_files = []

        for chunk in text_chunks:
            response = polly.synthesize_speech(Text=chunk, OutputFormat="mp3", VoiceId="Joanna")

            if "AudioStream" in response:
                logger.info("✅ Polly successfully generated audio.")
                audio_files.append(response["AudioStream"].read())  # ✅ FIXED INDENTATION HERE

        if not audio_files:
            raise Exception("No audio files generated.")

        def generate_audio():
            for audio in audio_files:
                yield audio

        return StreamingResponse(generate_audio(), media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"❌ TTS Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

import re

@app.post("/generate_phonics")
async def generate_phonics(request: QuestionRequest):
    try:
        logger.info("✅ Received a phonics processing request")

        phonics_prompt = f"""
        Take the following story and highlight phonics elements:
        - **Break words into syllables using dashes** (e.g., 'Ad-ven-ture').
        - **Highlight phonics_map sounds/words ONLY**.
        - **Return ONLY the cleaned story text—NO extra formatting, NO HTML, NO CSS.**
        - **Ensure the output is plain text without additional metadata.**

        Story:
        {request.story}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": phonics_prompt}]
        )

        # ✅ Step 1: Extract Response
        processed_story = response["choices"][0]["message"]["content"].strip()

        # ✅ Step 2: Clean Unwanted Formatting (Remove HTML, CSS, and Extra Tags)
        processed_story = re.sub(r"<.*?>", "", processed_story)  # Remove HTML tags
        processed_story = re.sub(r"\[.*?\]", "", processed_story)  # Remove bracketed text
        processed_story = re.sub(r"\{.*?\}", "", processed_story)  # Remove curly-brace content

        # ✅ Step 3: Log for Debugging
        logger.info(f"✅ Cleaned Phonics Story Output:\n{processed_story}")

        return {"phonicsStory": processed_story}

    except Exception as e:
        logger.error(f"❌ Error generating phonics story: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts_phonics")
async def generate_phonics_tts(request: dict):
    text = request.get("word", "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Missing word for pronunciation.")

    try:
        # ✅ Step 1: Split word into syllables for better enunciation
        syllables = re.sub(r"([aeiouy]+[^aeiouy]*)", r"\1<break time='250ms'/>", text, flags=re.IGNORECASE)

        # ✅ Step 2: Build SSML with slower speech and syllable breaks
        phonics_ssml = f"""
        <speak>
            <prosody rate="slow">
                {syllables}
            </prosody>
        </speak>
        """

        # ✅ Step 3: Log SSML for debugging
        logging.info(f"🔍 Phonics SSML Sent to Polly:\n{phonics_ssml}")

        # ✅ Step 4: Request TTS from Polly
        response = polly.synthesize_speech(
            Text=phonics_ssml,
            TextType="ssml",
            OutputFormat="mp3",
            VoiceId="Joanna"
        )

        # ✅ Step 5: Convert to audio stream and return
        audio_stream = response["AudioStream"].read()
        audio_buffer = BytesIO(audio_stream)

        return StreamingResponse(audio_buffer, media_type="audio/mpeg")

    except Exception as e:
        logging.error(f"❌ Polly TTS Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ✅ Start Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
