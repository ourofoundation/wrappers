import asyncio
import os

import requests
from dotenv import load_dotenv

from ouro import Ouro

load_dotenv()

STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")


async def poll_generation(generation_id: str, partial_file, user_id: str):
    """
    Polls the Stability API for the status of a video generation.
    """
    generation = None
    for _ in range(20):  # Poll 20 times
        await asyncio.sleep(10)  # Wait for 10 seconds between polls

        response = requests.get(
            f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}",
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
                "accept": "video/*",
            },
        )

        if response.status_code == 202:
            print("Generation in-progress, trying again in 10 seconds.")
            continue
        elif response.status_code == 200:
            print("Generation complete!")
            with open(f"generations/{generation_id}.mp4", "wb") as file:
                file.write(response.content)
            generation = {
                "partial_file": partial_file,
                "generation_id": generation_id,
                "content": response.content,
                "local_path": f"generations/{generation_id}.mp4",
            }
            break
        else:
            raise Exception(
                f"Generation failed with status code {response.status_code}"
            )

    # Once the generation is complete, upload the video
    await upload_video(generation, user_id)


async def upload_video(generation, user_id: str):
    """
    Uploads the generated video to Ouro, shares with the requesting user.
    """

    print(f"Generation {generation['generation_id']} completed. Uploading video...")

    partial_file = generation["partial_file"]
    local_file_path = generation["local_path"]

    # # Upload the video to Ouro
    ouro = Ouro(api_key=os.environ.get("OURO_API_KEY"))
    updated = ouro.files.update(
        id=partial_file.id, file_path=local_file_path, state="success"
    )
    print(f"Ouro file updated: {updated}")
