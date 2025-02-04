from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import requests
import boto3
import tempfile
import os
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI()

# Root route for testing
@app.get("/")
async def read_root():
    return {"message": "Backend is running"}

# Load environment variables from .env
load_dotenv()

# Configure Boto3 for Amazon Polly
polly_client = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
    endpoint_url=os.getenv("AWS_POLLY_ENDPOINT"),
)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set Pixabay API key
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local front-end development
        "https://yourstoryworld-b5butm9af-mohammed-abids-projects.vercel.app",  # Deployed front-end
    ],
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        body = await request.json()
        print(f"Incoming request body: {body}")
    except Exception as e:
        print(f"Failed to parse request body: {e}")
    response = await call_next(request)
    return response

# Define data model for story generation
class StoryRequest(BaseModel):
    genre: str
    characters: Optional[List[str]] = []
    setting: Optional[str] = ""
    themes: Optional[List[str]] = []
    length: str
    ageGroup: str
    storyDescription: Optional[str] = ""
    characterDetails: Optional[List[dict]] = []

@app.post("/generate")
async def generate_story(request: StoryRequest):
    try:
        country = "UK"  # Default to UK
        try:
            ip_response = requests.get("http://ip-api.com/json/", timeout=5)
            ip_response.raise_for_status()
            ip_data = ip_response.json()
            country = ip_data.get("countryCode", "UK")
            print(f"Detected country: {country}")
        except requests.exceptions.RequestException as e:
            print(f"IP detection failed with error: {e}. Defaulting to UK.")

        language = "British English" if country == "UK" else "American English"
        prompt = f"""Write a {request.length} {request.genre} story set in {request.setting or "any"}.
It should include the following characters: {', '.join(request.characters or [])}.
Themes: {', '.join(request.themes or [])}.
Description: {request.storyDescription or "any"}.
Character details: {request.characterDetails or "any"}.
Suitable for: {request.ageGroup} age group.

Provide a creative title for the story.

Guidelines:
1. Ensure the number of characters matches the list provided and refer to them consistently throughout the story.
2. Avoid introducing new characters, objects, or plot elements not explicitly mentioned in the input.
3. Keep the story child-friendly, adhering to the {request.ageGroup} age group.
4. Structure the story into clear, concise paragraphs suitable for young readers.
5. Use creative and engaging language to make the story interesting and imaginative.
6. Tailor the language, vocabulary, and tone to {language}.
7. End the story with a positive and uplifting resolution.
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a storytelling assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        full_story = response["choices"][0]["message"]["content"]
        title, story = full_story.split("\n", 1) if "\n" in full_story else ("Untitled", full_story)

        return {"title": title.strip(), "story": story.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_image")
async def generate_image(query: dict):
    try:
        pixabay_url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query['text']}&image_type=illustration&editors_choice=true&safesearch=true"
        response = requests.get(pixabay_url)

        if response.status_code != 200:
            raise Exception("Error fetching image from Pixabay API")

        data = response.json()
        if data.get("hits"):
            return {"image_url": data["hits"][0]["webformatURL"]}
        else:
            return {"image_url": "/images/default.png"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def text_to_speech(text: dict):
    try:
        print("Input text for TTS:", text["text"])
        response = polly_client.synthesize_speech(
            Text=text["text"],
            OutputFormat="mp3",
            VoiceId="Emma",
        )
        if "AudioStream" not in response or not response["AudioStream"]:
            raise Exception("AudioStream not present in Polly response.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(response["AudioStream"].read())
            audio_path = tmp_file.name

        print("Generated audio file path:", audio_path)
        return FileResponse(audio_path, media_type="audio/mpeg")
    except Exception as e:
        print("Error in TTS:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
