from typing import List, Dict, Optional
from dataclasses import dataclass

from magpy.objects.spline_handler import Spline

@dataclass
class Systematic:
    syst_name: str
    spline_name: str
    range: tuple[float, float]
    syst_type: str  # norm or spline
    modes: List[int]
    nominal: float
    error: float


    prior: str = "gaussian"  # gaussian, uniform

    fixed: bool = False

    # Oh yeah... it's readable
    kinematic_cuts: Optional[List[Dict[str, tuple[float, float]]]] = None
    is_circular: bool = False
    jump_around: Optional[float] = None

    def __post_init__(self):
        if self.kinematic_cuts is None:
            self.kinematic_cuts = [{}]

class SystematicHandler:
    def __init__(self, systematics: List[Systematic]):
        self._systematics = systematics
    
    @property
    def systematics(self) -> List[Systematic]:
        return self._systematics