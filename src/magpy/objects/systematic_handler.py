from typing import List, Dict, Optional

from dataclasses import dataclass


@dataclass
class Systematic:
    syst_name: str
    range: tuple[float, float]
    syst_type: str  # norm or spline
    modes: list[int]
    nominal: float
    error: float

    prior: str = "gaussian"  # gaussian, uniform

    fixed: bool = False

    # Oh yeah... it's readable
    kinematic_cuts: Optional[List[Dict[str, tuple[float, float]]]] = None

    def __post_init__(self):
        if self.kinematic_cuts is None:
            self.kinematic_cuts = [{}]


class SystematicHandler:
    def __init__(self, systematics: List[Systematic]):
        self.systematics = systematics
