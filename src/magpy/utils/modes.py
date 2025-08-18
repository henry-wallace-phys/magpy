from aenum import MultiValueEnum
from typing import Dict, List, TypedDict

class SplineModes(MultiValueEnum):
    CCQE = 0, "ccqe"
    CC2P2H = 1, "2p2h"
    CC1PIPM = 2, "cc1pipm"
    CCMPI = 3, "ccmultipi"
    CCDIS = 4, "ccdis"
    NCQE = 5, "ncqe"
    NC1PIPM = 6, "nc1pipm"
    NC1PI0 = 7, "nc1pi0"
    
    def __str__(self):
        return self.CC1PIPM
    
    def __int__(self):
        return self.values[0]
        
    def spline_name(self):
        return self.values[2]

    @classmethod
    def from_str(cls, mode_str: str):
        for mode in cls:
            if mode.values[1] == mode_str:
                return mode
        raise ValueError(f"Unknown mode: {mode_str}")

class SplineDict(TypedDict):
    bins: List[int]
    mode: str
    syst: str

def spline_name_to_bins(spline_name: str) -> SplineDict:
    '''
    Assume name of form dev.SYST.MODE.sp.BINX.BINY.BINZ;1
    '''

    # Get rid of backup info
    if ";" in spline_name:
        name_str = spline_name.split(";")
        name_str = name_str[0]

    name = name_str.split(".")
    
    s: SplineDict = {
        "syst": name[1],
        "mode": name[2],
        "bins": [int(n) for n in name[4:]]
    }

    return s

def bins_to_spline_name(syst: str, mode: str, bins: List[int]) -> str:
    '''
    Convert a systematic, mode and bins to a spline name
    '''
    return f"dev.{syst}.{mode}.sp." + ".".join(str(b) for b in bins)