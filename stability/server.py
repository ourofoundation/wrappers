# A wrapper around the StabilityAI API to organize it for use with Ouro

import base64
import json
import os
from io import BytesIO

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from lib.polling import poll_generation
from lib.schema import (
    ControlRequest,
    Fast3DRequest,
    ImageGenRequest,
    ImageToVideoRequest,
)
from ouro.utils import get_custom_openapi, ouro_field
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ouro import Ouro

load_dotenv()  # take environment variables from .env.

STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")


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




# Initialize FastAPI app
app = FastAPI(
    title="StabilityAI REST API",
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
if os.environ.get("PYTHON_ENV") != "development":
    app.add_middleware(BasicAuthMiddleware, token=os.environ.get("OURO_SERVICE_TOKEN") or "")


@app.post(
    "/stable-image/generate/ultra",
    summary="Text to image generation with Stable Image Ultra",
)
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "image")
async def generate_with_ultra(body: ImageGenRequest):
    try:
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/ultra",
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
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
async def generate_with_core(body: ImageGenRequest):
    try:
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
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
async def generate_with_sd3(body: ImageGenRequest):
    try:
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
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
async def control_with_sketch(body: ControlRequest):
    try:
        # Read the image file and pass it directly to next request
        file_response = requests.get(body.file.url)
        if file_response.status_code != 200:
            raise Exception("Failed to read the image file.")

        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/control/sketch",
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
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


@app.post(
    "/3d/stable-fast-3d",
    summary="Generate 3D assets from a single 2D input image",
    description="Stable Fast 3D generates high-quality 3D assets from a single 2D input image.",
)
@ouro_field("x-ouro-input-asset-type", "file")
@ouro_field("x-ouro-input-asset-filter", "image")
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "3d")
async def fast_3d(body: Fast3DRequest):
    try:
        # Read the image file and pass it directly to next request
        file_response = requests.get(body.file.url)
        if file_response.status_code != 200:
            raise Exception("Failed to read the image file.")

        response = requests.post(
            f"https://api.stability.ai/v2beta/3d/stable-fast-3d",
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
                "accept": "application/json",
            },
            files={"image": file_response.content},
            data={
                "texture_resolution": body.texture_resolution,
                "foreground_ratio": body.foreground_ratio,
            },
        )

        if response.status_code == 200:
            data = response.content
            # Encode the content as base64
            data = base64.b64encode(data).decode("utf-8")
        else:
            raise Exception(str(response.json()))

        return {
            "file": {
                "name": f"{body.file.name} as 3D model",
                "description": f"Generated model from an image using the StabilityAI API.",
                "base64": data,
                "type": "model/gltf-binary",
                "extension": "glb",
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/image-to-video",
    summary="Generate a short video from an image",
    description="Generate a short video based on an initial image with Stable Video Diffusion, a latent video diffusion model.",
)
@ouro_field("x-ouro-input-asset-type", "file")
@ouro_field("x-ouro-input-asset-filter", "image")
@ouro_field("x-ouro-output-asset-type", "file")
@ouro_field("x-ouro-output-asset-filter", "video")
# Flat rate of 20 credits per successful generation
async def image_to_video(
    body: ImageToVideoRequest,
    background_tasks: BackgroundTasks,
    # Get user id from ouro-user-id header
    ouro_user_id: str = Header(None),
):
    try:
        # Read the image file and pass it directly to next request
        file_response = requests.get(body.file.url)
        if file_response.status_code != 200:
            raise Exception("Failed to read the image file.")

        # Convert image to PNG format and handle resizing
        image = Image.open(BytesIO(file_response.content))
        # If square and greater than 768x768, resize to 768x768
        if image.width == image.height and image.width > 768:
            image = image.resize((768, 768), Image.Resampling.LANCZOS)
        # Always save as PNG with consistent settings
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG", optimize=True)
        content = img_byte_arr.getvalue()

        # Send the image to the Stability API
        response = requests.post(
            f"https://api.stability.ai/v2beta/image-to-video",
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
                "accept": "application/json",
            },
            files={"image": content},
            data={
                "seed": body.seed,
            },
        )
        if response.status_code != 200:
            raise Exception(str(response.json()["errors"]))

        generation_id = response.json().get("id")
        file_name = f"{body.file.name}.mp4"

        # Create partial file to later update with the generation
        ouro = Ouro(api_key=os.environ.get("OURO_API_KEY"))
        partial_file = ouro.files.create(
            name=file_name,
            visibility="public",
            metadata={"type": "video/mp4"},
            state="in-progress",
        )

        # Share with requesting user
        partial_file.share(ouro_user_id, "admin")
        # TODO: transfer ownership of the file to the user

        # Start up a background_task to check the status of the video generation
        background_tasks.add_task(
            poll_generation,
            generation_id,
            partial_file,
            ouro_user_id,
        )

        # Return to the user and tell them the video will be ready soon
        return Response(
            content=json.dumps(
                {
                    "message": "Video generation has been queued. The video will be shared with you soon.",
                    # "generation_id": generation_id,
                    "status": "pending",
                    "estimatedTime": "30 seconds",
                    "file": partial_file.model_dump(mode="json"),
                }
            ),
            media_type="application/json",
            status_code=status.HTTP_202_ACCEPTED,
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", port=8006, reload=True)
