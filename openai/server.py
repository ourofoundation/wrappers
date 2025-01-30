# A wrapper around the OpenAI API to organize it for use with Ouro

import base64
import json
import os
from base64 import b64encode
from io import BytesIO

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from lib.schema import (
    Dalle2ImageGenRequest,
    Dalle3ImageGenRequest,
    ImageAnalysisRequest,
    ImageEditRequest,
    ImageVariationRequest,
    PostToSpeechRequest,
    TextToSpeechRequest,
)
from openai import OpenAI
from ouro.utils import get_custom_openapi, ouro_field
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, token: str):
        super().__init__(app)
        self.token = token or None

    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        try:
            scheme, credentials = auth_header.split()
            print("Got credentials:", credentials)
            if scheme.lower() != "basic":
                raise HTTPException(
                    status_code=401, detail="Invalid authentication scheme"
                )
            if credentials != self.token:
                raise HTTPException(
                    status_code=401, detail="Invalid Ouro service credentials"
                )
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        return await call_next(request)


load_dotenv()  # take environment variables from .env.

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize FastAPI app
app = FastAPI(
    title="OpenAI API",
    description="OpenAI REST API provides a simple interface to state-of-the-art AI models for natural language processing, image generation, semantic search, and speech recognition.",
    servers=[
        {
            "url": "https://openai.ouro.foundation",
            "description": "Production environment",
        },
    ],
)

# Set the custom openapi function
app.openapi = get_custom_openapi(app, get_openapi)

# Allow origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.environ.get("OURO_FRONTEND_URL", "http://localhost:3000"),
        os.environ.get("OURO_BACKEND_URL", "http://localhost:8003"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if os.environ.get("PYTHON_ENV") != "development":
    app.add_middleware(
        BasicAuthMiddleware, token=os.environ.get("OURO_SERVICE_TOKEN") or ""
    )


@app.post(
    "/images/generations/dalle3",
    summary="Text to image generation with DALL-E 3",
)
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "image")
async def generate_with_dalle3(body: Dalle3ImageGenRequest):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=body.prompt,
            size=body.size.value,
            quality="standard",
            response_format="b64_json",
            n=1,
        )

        # Get the base64 image data
        image_data = response.data[0].b64_json

        return {
            "file": {
                "name": body.prompt[:50],
                "description": f'Generated image from "{body.prompt}" using DALL-E 3 from OpenAI.',
                "base64": image_data,
                "type": "image/png",
                "extension": "png",
            }
        }

    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/images/generations/dalle2",
    summary="Text to image generation with DALL-E 2",
)
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "image")
async def generate_with_dalle2(body: Dalle2ImageGenRequest):
    try:
        response = client.images.generate(
            model="dall-e-2",
            prompt=body.prompt,
            n=1,
            size=body.size.value,
            response_format="b64_json",
        )

        # Get the base64 image data
        image_data = response.data[0].b64_json

        return {
            "file": {
                "name": body.prompt[:50],
                "description": f'Generated image from "{body.prompt}" using DALL-E 2 from OpenAI.',
                "base64": image_data,
                "type": "image/png",
                "extension": "png",
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/images/variations",
    summary="Create variations of an existing image",
)
@ouro_field("x-ouro-input-asset-type", "file")
@ouro_field("x-ouro-input-asset-filter", "image")
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "image")
async def create_variation(body: ImageVariationRequest):
    try:
        # Read the image file
        file_response = requests.get(body.file.url)
        if file_response.status_code != 200:
            raise Exception("Failed to read the image file.")

        # Convert image to PNG and resize if needed
        image = Image.open(BytesIO(file_response.content))

        # OpenAI requires square images
        if image.width != image.height:
            size = min(image.width, image.height)
            left = (image.width - size) // 2
            top = (image.height - size) // 2
            image = image.crop((left, top, left + size, top + size))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save to BytesIO
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        response = client.images.create_variation(
            image=img_byte_arr, n=1, size="1024x1024", response_format="b64_json"
        )

        image_data = response.data[0].b64_json

        return {
            "file": {
                "name": f"Variation of {body.file.name}",
                "description": "Generated image variation using DALL-E.",
                "base64": image_data,
                "type": "image/png",
                "extension": "png",
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/images/analyze",
    summary="Use vision capabilities to understand images",
)
@ouro_field("x-ouro-input-asset-type", "file")
@ouro_field("x-ouro-input-asset-filter", "image")
async def analyze_image(body: ImageAnalysisRequest):
    try:
        # Read the image file
        file_response = requests.get(body.file.url)
        if file_response.status_code != 200:
            raise Exception("Failed to read the image file.")

        # Convert the image to base64
        image_base64 = b64encode(file_response.content).decode("utf-8")

        # Create the messages for the API
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": body.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=body.max_tokens,
        )

        return {"analysis": response.choices[0].message.content}

    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/speech/generate",
    summary="Convert text to speech using OpenAI TTS",
)
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "audio")
async def generate_speech(body: TextToSpeechRequest):
    try:
        response = client.audio.speech.create(
            model=body.model,
            voice=body.voice.value,
            input=body.text,
        )

        # Get the audio data
        audio_data = base64.b64encode(response.content).decode("utf-8")

        # Create a preview of the text if it's too long
        text_preview = (
            body.text[:20] + "..." if len(body.text) > 50 else body.text
        ).replace("\n", "")

        return {
            "file": {
                "name": f"{text_preview} as audio with the {body.voice.value} voice",
                "description": f"Generated speech from text using {body.voice.value} voice.",
                "base64": audio_data,
                "type": "audio/mp3",
                "extension": "mp3",
            }
        }

    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/speech/from-post",
    summary="Convert a post to speech using OpenAI TTS",
)
@ouro_field("x-ouro-input-asset-type", "post")
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "audio")
async def generate_speech_from_post(body: PostToSpeechRequest):
    try:
        text_content = body.post.content["text"]
        # Generate speech from the post content
        response = client.audio.speech.create(
            model=body.model,
            voice=body.voice.value,
            input=text_content,
        )
        # Get the audio data
        audio_data = base64.b64encode(response.content).decode("utf-8")
        # Create a preview of the text
        text_preview = (
            text_content[:20] + "..." if len(text_content) > 50 else text_content
        ).replace("\n", "")

        return {
            "file": {
                "name": f"{text_preview} as audio with the {body.voice.value} voice",
                "description": f"Generated speech from post using {body.voice.value} voice from OpenAI.",
                "base64": audio_data,
                "type": "audio/mp3",
                "extension": "mp3",
            }
        }

    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", port=8020, reload=True)
