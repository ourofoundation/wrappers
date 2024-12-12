# A wrapper around the LumaAI API to organize it for use with Ouro

import base64
import json
import os
import re
from enum import Enum
from typing import List, Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from lib.client import client
from lib.polling import poll_generation
from lib.schema import VideoGenRequest
from lumaai import LumaAI
from ouro.utils import get_custom_openapi, ouro_field
from pydantic import BaseModel, Field

from ouro import Ouro

load_dotenv()  # take environment variables from .env.

# Initialize FastAPI app
app = FastAPI(
    title="Dream Machine API",
    version="v1.0.2",
    description="Build and scale creative products with the world's most popular and intuitive video generation models using the Dream Machine API",
    servers=[
        {
            "url": "https://luma.ouro.foundation",
            "description": "Production environment",
        },
    ],
    terms_of_service="",
    contact={
        "name": "",
        "url": "",
        "email": "",
    },
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


@app.post(
    "/dream-machine/generate/text-to-video",
    summary="Text to video generation with Dream Machine",
)
# @ouro_field("x-ouro-output-asset-type", "file")
# @ouro_field("x-ouro-output-asset-filter", "video")
async def generate_video(
    request: VideoGenRequest,
    background_tasks: BackgroundTasks,
    # Get user id from ouro-user-id header
    ouro_user_id: str = Header(None),
):
    try:
        generation = client.generations.create(
            prompt=request.prompt,
            aspect_ratio=request.aspect_ratio,
            loop=request.loop,
            # enhance_prompt=request.enhance_prompt,
        )

        if generation.failure_reason:
            raise Exception(generation.failure_reason)

        # 3723f484-01df-4950-93af-e7c159945766

        # Start up a background_task to check the status of the video generation
        background_tasks.add_task(
            poll_generation,
            generation.id,
            request,
            ouro_user_id,
        )

        # If successful, return success to the user and tell them the video will be ready soon
        return Response(
            content=json.dumps(
                {
                    "note": "Video generation has been queued. The video will be ready in a few minutes.",
                    "id": generation.id,
                    "state": generation.state,
                }
            ),
            media_type="application/json",
            status_code=status.HTTP_201_CREATED,
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", port=8011, reload=True)
