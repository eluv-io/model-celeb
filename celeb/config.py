from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RuntimeConfig:
    fps: float
    thres: float
    min_box_size: float
    ipt_rgb: bool
    allow_single_frame: bool
    ground_truth: str
    content_type: str
    content_id: Optional[str] = None
    restrict_list: Optional[List[str]] = None