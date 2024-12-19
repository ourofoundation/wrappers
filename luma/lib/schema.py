from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AspectRatioEnum(str, Enum):
    ratio_4_3 = "4:3"
    ratio_16_9 = "16:9"


class VideoGenRequest(BaseModel):
    prompt: str = Field(
        ...,
        title="Prompt",
        description="What you wish to see in the output video.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        AspectRatioEnum.ratio_16_9,
        title="Aspect Ratio",
        description="Controls the aspect ratio of the generated video.",
    )
    loop: bool = Field(
        False,
        title="Loop",
        description="Whether the output video should loop.",
    )
    enhance_prompt: bool = Field(
        True,
        title="Enhance Prompt",
        description="Whether to automatically generate additional descriptions to help create a more detailed generation.",
    )
