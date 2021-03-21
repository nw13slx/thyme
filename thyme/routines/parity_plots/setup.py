import colorsys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

plt.switch_backend("agg")

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 8

tabcolors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
darkcolors = [colorsys.rgb_to_hls(*colors.to_rgb(c)) for c in tabcolors]
amount = 1.1
darkcolors = [
    [colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])] for c in darkcolors
]
