"""Shared grayscale figure style: serif text, black/grey lines, hatching instead of colour, legends never over data."""
from config import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT + '/work/figs'
GREYS = ['#000000', '#4d4d4d', '#7f7f7f', '#a6a6a6', '#cccccc', '#e6e6e6']
HATCH = ['', '///', '...', 'xxx', '\\\\\\', '++', 'oo']
MARK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
LS = ['-', '--', ':', '-.', (0, (5, 1)), (0, (1, 1))]


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
        'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
        'legend.fontsize': 6.6, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'lines.linewidth': 1.0, 'axes.grid': True, 'grid.color': '#d9d9d9', 'grid.linewidth': 0.4,
        'hatch.linewidth': 0.5, 'legend.frameon': True, 'legend.edgecolor': '#808080', 'legend.framealpha': 1.0,
        'savefig.dpi': 450, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.02, 1.04, f'({letter})', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, fontweight='bold')
