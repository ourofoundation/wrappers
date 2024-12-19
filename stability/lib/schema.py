from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


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


class TextureResolutionEnum(str, Enum):
    res_1024 = 1024
    res_2048 = 2048
    res_512 = 512


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


class Fast3DRequest(BaseModel):
    file: File
    texture_resolution: TextureResolutionEnum = Field(
        1024,
        title="Texture Resolution",
        description="Determines the resolution of the textures used for both the albedo (color) map and the normal map.",
    )
    foreground_ratio: float = Field(
        0.85,
        title="Foreground Ratio",
        description="Controls the amount of padding around the object to be processed within the frame. A higher ratio means less padding and a larger object, while a lower ratio increases the padding.",
        ge=0,
        le=1,
    )


class ImageToVideoRequest(BaseModel):
    file: File
    seed: int = Field(
        42,
        title="Seed",
        description="A seed value to use for the video generation.",
    )
