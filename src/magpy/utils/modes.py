from aenum import MultiValueEnum
from typing import Sequence, List, TypedDict

from magpy.Exceptions import MagpyModeNotFoundException

class SplineModes(MultiValueEnum):
    CCQE = 0, "ccqe", (1,)
    CC2P2H = 1, "2p2h", (2,)
    CC1PIPM = 2, "cc1pipm", (11, 12, 13)
    CCMPI = 3, "ccmultipi", (21,)
    CCDIS = 4, "ccdis", (26,)
    NCQE = 5, "ncqe", (51, 52)
    NC1PIPM = 6, "nc1pipm", (33, 34)
    NC1PI0 = 7, "nc1pi0", (31, 32)
    
    
    def spline_name(self):
        return self.values[1]

    @classmethod
    def from_str(cls, mode_str: str):
        for mode in cls:
            if mode.values[1] == mode_str:
                return mode
        raise ValueError(f"Unknown mode: {mode_str}")

    @classmethod
    def from_generator_mode(cls, gen: int):
        for mode in cls:
            if abs(gen) in mode.values[2]:
                return mode
        raise MagpyModeNotFoundException(f"Unknown generator mode: {gen}")

class SplineDict(TypedDict):
    bins: Sequence[int]
    mode: str
    syst: str


def spline_name_to_bins(spline_name: str) -> SplineDict:
    """
    Assume name of form dev.SYST.MODE.sp.BINX.BINY.BINZ;1
    """

    # Get rid of backup info
    if ";" in spline_name:
        name_str = spline_name.split(";")
        name_str = name_str[0]

    name = name_str.split(".")

    s: SplineDict = {
        "syst": name[1],
        "mode": name[2],
        "bins": [int(n) for n in name[4:]],
    }

    return s


def bins_to_spline_name(syst: str, mode: str, bins: List[int]) -> str:
    """
    Convert a systematic, mode and bins to a spline name
    """
    return f"dev.{syst}.{mode}.sp." + ".".join(str(b) for b in bins)
