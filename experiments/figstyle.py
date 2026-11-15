"""Estilo único das figuras: coluna IEEE (3.45 pol), serif, paleta Okabe-Ito (segura para daltonismo), sem moldura superior/direita."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#6b6b6b", "#56B4E9"
W = 3.45
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42, "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.4, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": .5, "axes.axisbelow": True, "lines.linewidth": 1.2, "lines.markersize": 3.6, "axes.linewidth": .7,
    "xtick.major.width": .7, "ytick.major.width": .7, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": .03})
def panel(ax, s): ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
def zero(ax, y=0.0): ax.axhline(y, color="#222", lw=.8, zorder=1)
