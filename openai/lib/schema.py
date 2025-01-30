from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Dalle3SizeEnum(str, Enum):
    square = "1024x1024"
    tall = "1024x1792"
    wide = "1792x1024"


class Dalle2SizeEnum(str, Enum):
    square_small = "256x256"
    square_medium = "512x512"
    square_large = "1024x1024"


class FileInfo(BaseModel):
    url: str
    name: str


class Dalle3ImageGenRequest(BaseModel):
    prompt: str = Field(
        ...,
        title="Prompt",
        description="The text prompt to generate the image from.",
    )
    size: Dalle3SizeEnum = Field(
        "1024x1024",
        title="Resolution",
        description="Controls the resolution of the generated image.",
    )


class Dalle2ImageGenRequest(BaseModel):
    prompt: str = Field(
        ...,
        title="Prompt",
        description="The text prompt to generate the image from.",
    )
    size: Dalle2SizeEnum = Field(
        "1024x1024",
        title="Resolution",
        description="Controls the resolution of the generated image.",
    )


class File(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    url: str
    filename: str
    type: str


class Post(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None

    class Config:
        extra = "allow"  # Allows additional fields to pass through


class ImageToVideoRequest(BaseModel):
    file: File
    seed: int = Field(
        42,
        title="Seed",
        description="A seed value to use for the video generation.",
    )


class ImageVariationRequest(BaseModel):
    file: FileInfo


class ImageEditRequest(BaseModel):
    file: FileInfo
    prompt: str = Field(
        ...,
        title="Prompt",
        description="The text prompt describing the desired modifications to the image.",
    )
    mask: Optional[FileInfo] = None


class ImageAnalysisRequest(BaseModel):
    file: FileInfo
    prompt: str = Field(
        ...,
        title="Prompt",
        description="The question or prompt about the image you want to analyze.",
    )
    max_tokens: int = Field(
        300,
        title="Max Tokens",
        description="The maximum number of tokens to generate in the response.",
    )


class VoiceEnum(str, Enum):
    alloy = "alloy"
    ash = "ash"
    coral = "coral"
    echo = "echo"
    fable = "fable"
    onyx = "onyx"
    nova = "nova"
    sage = "sage"
    shimmer = "shimmer"


class VoiceModelEnum(str, Enum):
    tts_1 = "tts-1"
    tts_1_hd = "tts-1-hd"


class TextToSpeechRequest(BaseModel):
    text: str = Field(
        ...,
        title="Text",
        description="The text to convert to speech",
    )
    voice: VoiceEnum = Field(
        VoiceEnum.alloy,
        title="Voice",
        description="The voice to use for the speech",
    )
    model: VoiceModelEnum = Field(
        VoiceModelEnum.tts_1,
        title="Model",
        description="The TTS model to use",
    )


class PostToSpeechRequest(BaseModel):
    post: Post = Field(
        ...,
        title="Post",
        description="The post to convert to speech",
    )
    voice: VoiceEnum = Field(
        VoiceEnum.alloy,
        title="Voice",
        description="The voice to use for the speech",
    )
    model: VoiceModelEnum = Field(
        VoiceModelEnum.tts_1,
        title="Model",
        description="The TTS model to use",
    )
