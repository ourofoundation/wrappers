# A wrapper around the StabilityAI API to organize it for use with Ouro

import json
import os
import re
from enum import Enum
from typing import List, Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from ouro.utils import get_custom_openapi, ouro_field
from pydantic import BaseModel, Field

from ouro import Ouro

load_dotenv()  # take environment variables from .env.


class AspectRatioEnum(str, Enum):
    square = "1:1"
    ratio_16_9 = "16:9"
    ratio_21_9 = "21:9"
    ratio_2_3 = "2:3"
    ratio_3_2 = "3:2"
    ratio_4_5 = "4:5"
    ratio_5_4 = "5:4"
    ratio_9_16 = "9:16"
    ratio_9_21 = "9:21"


class ImageGenRequest(BaseModel):
    prompt: str = Field(
        ...,
        title="Prompt",
        description="What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results.",
    )
    negative_prompt: Optional[str] = Field(
        None,
        title="Negative Prompt",
        description="A blurb of text describing what you do not wish to see in the output image.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        "1:1",
        title="Aspect Ratio",
        description="Controls the aspect ratio of the generated image.",
    )
    # output_format: str = Field(
    #     "png",
    #     title="Output Format",
    #     description="Dictates the content-type of the generated image.",
    # )


class File(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    url: str
    filename: str
    type: str


class ControlRequest(BaseModel):
    file: File
    prompt: str = Field(
        ...,
        title="Prompt",
        description="What you wish to see in the output image. A strong, descriptive prompt that clearly defines elements, colors, and subjects will lead to better results.",
    )
    control_strength: float = Field(
        0.7,
        title="Control Strength",
        description="How much influence, or control, the image has on the generation. Represented as a float between 0 and 1, where 0 is the least influence and 1 is the maximum.",
        ge=0,
        le=1,
    )
    negative_prompt: Optional[str] = Field(
        None,
        title="Negative Prompt",
        description="A blurb of text describing what you do not wish to see in the output image.",
    )
    # output_format: str = Field(
    #     "png",
    #     title="Output Format",
    #     description="Dictates the content-type of the generated image.",
    # )


# Initialize FastAPI app
app = FastAPI(
    title="StabilityAI API",
    version="v2beta",
    description="",
    servers=[
        {
            "url": "https://stability.ouro.foundation",
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
    "/stable-image/generate/ultra",
    summary="Text to image generation with Stable Image Ultra",
)
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "image")
async def generate_with_ultra(
    body: ImageGenRequest, authorization: str | None = Header(default=None)
):
    try:
        api_key = authorization.split(" ")[1] if authorization else None
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/ultra",
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "application/json",
            },
            files={"none": ""},
            data={
                "prompt": body.prompt,
                "negative_prompt": body.negative_prompt,
                "output_format": "png",
            },
        )

        if response.status_code == 200:
            data = response.json()
            # Assert we finished with data
            assert data.get("image"), "No image data in response."
            assert data["finish_reason"] == "SUCCESS", "Failed to generate image."

        else:
            raise Exception(str(response.json()))

        return {
            "file": {
                "name": body.prompt,
                "description": f'Generated image from "{body.prompt}" using the StabilityAI API.',
                "base64": data["image"],
                "type": "image/png",
                "extension": "png",
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/stable-image/generate/core",
    summary="Text to image generation with Stable Image Core",
)
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "image")
async def generate_with_core(
    body: ImageGenRequest, authorization: str | None = Header(default=None)
):
    try:
        api_key = authorization.split(" ")[1] if authorization else None
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "application/json",
            },
            files={"none": ""},
            data={
                "prompt": body.prompt,
                "negative_prompt": body.negative_prompt,
                "output_format": "png",
            },
        )

        if response.status_code == 200:
            data = response.json()
            # Assert we finished with data
            assert data.get("image"), "No image data in response."
            assert data["finish_reason"] == "SUCCESS", "Failed to generate image."

        else:
            raise Exception(str(response.json()))

        return {
            "file": {
                "name": body.prompt,
                "description": f'Generated image from "{body.prompt}" using the StabilityAI API.',
                "base64": data["image"],
                "type": "image/png",
                "extension": "png",
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/stable-image/generate/sd3",
    summary="Text to image generation with Stable Diffusion 3.0",
)
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "image")
async def generate_with_sd3(
    body: ImageGenRequest, authorization: str | None = Header(default=None)
):
    try:
        api_key = authorization.split(" ")[1] if authorization else None
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "application/json",
            },
            files={"none": ""},
            data={
                "prompt": body.prompt,
                "negative_prompt": body.negative_prompt,
                "output_format": "png",
            },
        )

        if response.status_code == 200:
            data = response.json()
            # Assert we finished with data
            assert data.get("image"), "No image data in response."
            assert data["finish_reason"] == "SUCCESS", "Failed to generate image."

        else:
            raise Exception(str(response.json()))

        return {
            "file": {
                "name": body.prompt,
                "description": f'Generated image from "{body.prompt}" using the StabilityAI API.',
                "base64": data["image"],
                "type": "image/png",
                "extension": "png",
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/stable-image/control/sketch",
    summary="Image-to-image generation for controlled variations of existing images or sketches",
)
@ouro_field("x-ouro-input-asset-type", "file")
@ouro_field("x-ouro-input-asset-filter", "image")
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "image")
async def control_with_sketch(
    body: ControlRequest, authorization: str | None = Header(default=None)
):
    try:
        # Read the image file and pass it directly to next request
        file_response = requests.get(body.file.url)
        if file_response.status_code != 200:
            raise Exception("Failed to read the image file.")

        api_key = authorization.split(" ")[1] if authorization else None
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/control/sketch",
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "application/json",
            },
            files={"image": file_response.content},
            data={
                "prompt": body.prompt,
                "control_strength": body.control_strength,
                "negative_prompt": body.negative_prompt,
                "output_format": "png",
            },
        )

        if response.status_code == 200:
            data = response.json()
            # Assert we finished with data
            assert data.get("image"), "No image data in response."
            assert data["finish_reason"] == "SUCCESS", "Failed to generate image."

        else:
            raise Exception(str(response.json()))

        return {
            "file": {
                "name": body.prompt,
                "description": f'Generated image from "{body.prompt}" using the StabilityAI API.',
                "base64": data["image"],
                "type": "image/png",
                "extension": "png",
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", port=8006, reload=True)
