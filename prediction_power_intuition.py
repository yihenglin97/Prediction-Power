"""
Make sure that LaTeX is installed:

sudo apt update
sudo apt install texlive texlive-latex-extra texlive-fonts-extra dvipng cm-super
pip install latex
"""
import matplotlib.pyplot as plt
import numpy as np

# hide top and right splines on plots
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

rc_fonts = {
    "font.family": "serif",
    "font.size": 12,
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{times}\usepackage{amsfonts}',
}
plt.rcParams.update(rc_fonts)

u = np.linspace(-2, 2, num=100)

# baseline
EQ0 = u**2 + 1

# with perfect predictors
EQ_vpos1 = (u+1)**2
EQ_vneg1 = (u-1)**2

fig, ax = plt.subplots(1, 1, figsize=(6.5, 2), tight_layout=True)
ax.plot(u, EQ0,
        color='tab:blue',
        label=r'$\mathrm{\mathbb{E}}[Q_{T-1}^{\bar\pi}(0,u; \Xi) \mid V_{T-1}(\mathbf{0})]$')
ax.plot(0, 1, zorder=10, marker='o', color='tab:blue')
ax.plot(u, EQ_vneg1,
        color='tab:orange',
        label=r'$\mathrm{\mathbb{E}}[Q_{T-1}^{\pi^\theta}(0,u; \Xi) \mid V_{T-1}(\theta) = -1]$')
ax.plot(1, 0, zorder=10, marker='o', color='tab:orange')
ax.plot(u, EQ_vpos1,
        color='tab:green',
        label=r'$\mathrm{\mathbb{E}}[Q_{T-1}^{\pi^\theta}(0,u; \Xi) \mid V_{T-1}(\theta) = 1]$')
ax.plot(-1, 0, zorder=10, marker='o', color='tab:green')
ax.set(xlabel='$u$', ylabel='Q value', ylim=(-0.4, 6))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

fig.savefig('Figures/prediction_improvement_intuition.pdf', pad_inches=0, bbox_inches='tight')
