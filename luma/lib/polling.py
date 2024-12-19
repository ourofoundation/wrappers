import asyncio
import os

import requests

from ouro import Ouro

from .client import client
from .schema import VideoGenRequest

ouro = Ouro(
    api_key=os.environ.get("OURO_API_KEY"),
)


async def poll_generation(generation_id: str, request: VideoGenRequest, user_id: str):
    """
    Polls the Luma API for the status of a video generation.
    """
    # Simulating polling a resource
    generation = None
    for _ in range(20):  # Poll 10 times
        await asyncio.sleep(10)  # Wait for 10 seconds between polls

        generation = client.generations.get(generation_id)
        print(f"Generation {generation.id} is {generation.state}")
        if generation.state == "completed":
            break

    # If polling fails, raise an error
    if not generation or generation.state != "completed":
        raise Exception("Generation failed")

    # Once the generation is complete, upload the video
    await upload_video(generation, request, user_id)


async def upload_video(generation, request: VideoGenRequest, user_id: str):
    """
    Uploads the generated video to Ouro, shares with the requesting user.
    """

    print(f"Generation {generation.id} completed. Uploading video...")

    url = generation.assets.video
    response = requests.get(url, stream=True)

    file_name = f"{request.prompt}.mp4"
    with open(file_name, "wb") as file:
        file.write(response.content)
    print(f"File downloaded as {file_name}")

    # Upload the video to Ouro
    ouro.files.create(file_name, name=request.prompt, visibility="public")
