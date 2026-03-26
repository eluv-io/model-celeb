from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RuntimeConfig:
    thres: float = 0.4
    min_box_size: float = 0
    ground_truth: str = "IBC"
    content_type: str = "video"
    content_id: Optional[str] = None
    restrict_list: Optional[List[str]] = None