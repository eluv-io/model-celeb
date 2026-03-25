from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RuntimeConfig:
    fps: float = 4
    thres: float = 0.4
    min_box_size: float = 0
    allow_single_frame: bool = False
    ground_truth: str = "IBC"
    content_type: str = "video"
    content_id: Optional[str] = None
    restrict_list: Optional[List[str]] = None

    continue_on_error: bool = False