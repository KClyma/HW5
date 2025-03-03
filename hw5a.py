# region problem statement
"""
BYOMD (Build Your Own Moody Diagram). The Darcy-Weisbach friction factor (f) is used to compute head loss in pipe flow through:

ℎ𝑓=𝑓𝐿𝐷𝑉²/2𝑔

where L=pipe length, D = pipe diameter, V = average velocity of the fluid, and g is the acceleration of gravity.

f is known to vary with both Reynolds number (Re) and pipe wall roughness (Relative roughness (ϵ/d)).
In the laminar range (Re<2000), the relative roughness seems to be irrelevant where 𝑓=64/𝑅𝑒, whereas in the turbulent range (Re>4000), f is described by the empirical and implicit Colebrook equation:

1/𝑓¹/² = −2.0 log(𝜖/𝑑 / 3.7 + 2.51 / 𝑅𝑒⋅𝑓¹/²)

We note that in the Colebrook equation, f cannot be found analytically, so we must use an iterative method (i.e., fsolve) to find f at each Re and ϵ/d coordinate. At intermediate Re, the flow is called transitional, and f is not easily predicted due to instability.

The Moody diagram graphically displays f as a function of Re and ϵ/d for a finite set of relative roughness. Write a program that produces a Moody diagram that has all the features like the one below.
"""
# endregion

# region imports
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np  # Ensuring numpy is used in the script


# endregion

# region functions
def ff(Re: float, rr: float, CBEQN: bool = False) -> float:
    """
    Calculates the Darcy-Weisbach friction factor (f) for a given Reynolds number (Re) and relative roughness (ϵ/d).

    - Uses **Colebrook equation** for turbulent flow (Re > 4000).
    - Uses **laminar equation** (f = 64/Re) for Re < 2000.
    - Interpolates friction factor for **transitional flow (2000 ≤ Re ≤ 4000)**.

    :param Re: Reynolds number (unitless)
    :param rr: Relative roughness (ϵ/d, between 0 and 0.05)
    :param CBEQN: Boolean for using Colebrook equation (True for turbulent flow)
    :return: Computed friction factor (f)
    """
    if CBEQN:
        # Colebrook equation: Solve for f using fsolve
        cb = lambda f: 1 / np.sqrt(abs(f)) + 2.0 * np.log10(rr / 3.7 + 2.51 / (Re * np.sqrt(abs(f))))
        result = fsolve(cb, max(0.02, 1e-5))  # Ensure positive initial guess
        return result[0]
    else:
        return 64 / Re  # Laminar flow equation
    pass

def plotMoody(plotPoint=False, pt=(0,0), marker='o', markersize=10, markeredgecolor='red', markerfacecolor='none', label=None, clear_plot=False):
    """
    This function produces the Moody diagram for a Re range from 1 to 10^8 and
    for relative roughness from 0 to 0.05 (20 steps). The laminar region is described
    by the simple relationship of f=64/Re whereas the turbulent region is described by
    the Colebrook equation.
    :return: just shows the plot, nothing returned
    """
    # Step 1: create logspace arrays for ranges of Re
    ReValsCB = np.logspace(np.log10(4000), np.log10(1e8), 100)   # turbulent flow (Re 4000 to 10^8)
    ReValsL = np.logspace(np.log10(600.0), np.log10(2000.0), 20)  # Laminar flow (Re 600 to 2000)
    ReValsTrans = np.logspace(np.log10(2000), np.log10(4000), 50)  # transition flow (Re 2000 to 4000)

    # Step 2: create array for range of relative roughnesses
    rrVals = np.array([0, 1E-6, 5E-6, 1E-5, 5E-5, 1E-4, 2E-4, 4E-4, 6E-4, 8E-4, 1E-3, 2E-3, 4E-3, 6E-3,
                       8E-3, 1.5E-2, 2E-2, 3E-2, 4E-2, 5E-2])

    # Step 2: calculate the friction factor in the laminar range
    ffLam = np.array([64/Re for Re in ReValsL])  # Compute f = 64/Re for all Re in laminar range
    ffTrans = np.array([ff(Re, 0) for Re in ReValsTrans])  # Compute transition region friction factor

    # Step 3: calculate friction factor values for each rr at each Re for turbulent range
    ffCB = np.array([[ff(Re, relRough, CBEQN=True) for Re in ReValsCB] for relRough in rrVals])  #for relRough in rrVals])

    # Step 5:  construct the plot
    plt.loglog(ReValsL, ffLam, 'b-', linewidth=1.5, label='Laminar')   #plot the laminar part as a solid line
    plt.loglog(ReValsTrans, ffTrans, 'r--', linewidth=1.5, label='Transition')  #plot transition part as dashed line
    for nRelR in range(len(ffCB)):
        plt.loglog(ReValsCB, ffCB[nRelR], 'k',linewidth=1) #plot turbulent region for each pipe roughness
        plt.annotate(xy=(ReValsCB[-1], ffCB[nRelR][-1]),text=f"{rrVals[nRelR]:.2e}")  #annotate each curve

    plt.xlim(600, 1E8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number $Re$", fontsize=16)  #Correct axis label
    plt.ylabel(r"Friction factor $f$", fontsize=16)  #Correct axis label
    plt.text(2.5E8, 0.02, r"Relative roughness $\frac{\epsilon}{d}$", rotation=90, fontsize=16)

    ax = plt.gca()  # capture the current axes for use in modifying ticks, grids, etc.
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)  # format tick marks
    ax.tick_params(axis='both', grid_linewidth=1, grid_linestyle='solid', grid_alpha=0.5)
    ax.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.grid(which='both')
    if plotPoint:
        plt.plot(pt[0], pt[1], marker=marker, markersize=markersize, markeredgecolor=markeredgecolor,
                 markerfacecolor=markerfacecolor, label=label)  # plot the point with specified properties

    plt.show()

def main():
    plotMoody()
# endregion


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion