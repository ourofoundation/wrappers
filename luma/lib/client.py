import os

from dotenv import load_dotenv
from lumaai import LumaAI

load_dotenv()  # take environment variables from .env.

client = LumaAI(
    auth_token=os.environ.get("LUMAAI_API_KEY"),
)
