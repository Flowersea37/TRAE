from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional


MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
# MAX_PIXELS = 262144


class DatasetType(Enum):
    MATHVISTA = "mathvista"
    MATHVERSE = "mathverse"
    MATHVISION = "mathvision"
    MATHVISION_ALL = "mathvision_all"
    MATHVISTA_GPS = "mathvista_gps"
    EMMA_MATH = "emma-math"
    EMMA_CHEM = "emma-chem"
    EMMA_CODE = "emma-code"
    EMMA_PHYSICS = "emma-physics"
    MMMU_PRO_10 = "mmmu-pro-10"
    MMMU_PRO_4 = "mmmu-pro-4"
    MMMU_PRO_VISION = "mmmu-pro-vision"

@dataclass
class DatasetConfig:
    name: str
    split: str
    image_field: str
    image_url_field: str
    response_field: str
    instruction_field: Optional[str] = None
    subset: Optional[str] = None
    choices_field: Optional[str] = None
    options_field: Optional[str] = None
    source_field: Optional[str] = None


@dataclass
class ModelConfig:
    model_name: str
    num_sequence: int = 1
    out_seq_length: int = 2048
    top_p: float = 0.001
    top_k: int = 1
    temperature: float = 0.01
    repetition_penalty: float = 1.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    greedy: str = 'false'
    presence_penalty: float = 1.5
    enforce_eager: bool = True
    # limit_mm_per_prompt: dict = field(
    #     default_factory=lambda: {
    #         "image": 1,
    #         "video": 0,
    #         "audio": 0,
    #     }
    # ) 