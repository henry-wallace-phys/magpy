"""
Really simple TSpline3 converter, not designed to be used in the magpy ecosystem
"""

import ROOT
from tqdm import tqdm

"""
Converts to format accessible to magpy
"""


def tspline3_converter(input_file: str, input_tree: str = "sample_sum"):
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


if __name__ == "__main__":
    FILE = "/Users/henrywallace/software/MaCh3/Tutorial/src/TutorialConfigs/MC/SplineFile.root"
    tspline3_converter(FILE)
