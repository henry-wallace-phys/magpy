"""
Really simple TSpline3 converter, not designed to be used in the magpy ecosystem
"""

import ROOT
from tqdm import tqdm
import ctypes

"""
Converts to format accessible to magpy
"""


def sample_sum_tspline3_converter(input_file: str, input_tree: str = "sample_sum"):
    input_frame = ROOT.RDataFrame(input_tree, input_file)

    # Extract graphs
    frame_as_numpy = input_frame.AsNumpy()

    graph_dict = {}
    for key, value in frame_as_numpy.items():
        graph_dict[key] = [v[0] for v in value]

    # Save each graph to a ROOT file
    converted_file = ROOT.TFile("converted_splines.root", "RECREATE")
    converted_file.cd()

    print("Converting to file")
    for syst, splines in tqdm(
        graph_dict.items(), desc="Systematic", total=len(graph_dict)
    ):
        converted_file.mkdir(syst)
        converted_file.cd(syst)

        for i, spline in enumerate(splines):
            spline.Write(f"{spline.GetName()}_{i}")

def tspline3_converter(input_file_name: str):
    input_file = ROOT.TFile(input_file_name, "READ")
    out_file = ROOT.TFile("converted_splines.root", "RECREATE")
    
    # Loop over objects in file
    out_file.cd()
    for key in tqdm(input_file.GetListOfKeys(), desc="Converting TSpline3"):
        if not key.GetClassName() == "TSpline3":
            continue
        spline: ROOT.TSpline3 = key.ReadObj()
        out_graph = ROOT.TGraph(spline.GetNp())

        for i in range(spline.GetNp()):
            x=ctypes.c_double()
            y=ctypes.c_double()
            b=ctypes.c_double()
            c=ctypes.c_double()
            d=ctypes.c_double()
            spline.GetCoeff(i,x,y, b,c,d)
            out_graph.SetPoint(i, x.value, y.value)

        out_graph.Write(f"{spline.GetName()}")

if __name__ == "__main__":
    FILE = "/Users/henrywallace/software/magpy/BinnedSplinesTutorialInputs2D.root"
    tspline3_converter(FILE)
