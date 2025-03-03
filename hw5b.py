# region problem statement
"""
b. Beyond BYOMD:
Create a program that solicits input from the user for:
    - Pipe diameter in inches
    - Pipe roughness (ϵ) in micro-inches or mics (10⁻⁶ inches)
    - Flow rate in gallons/min

The program then returns the head loss per foot (hf/L) in appropriate English units.

Furthermore, you should display the Moody diagram with an icon:
    - An upward triangle if the flow is in the transition range (2000 ≤ Re ≤ 4000)
    - A circle if otherwise.

Your program should allow the user to re-specify the parameters and keep track of each `f` by adding a new icon to the Moody diagram with each new set of parameters.

Note: It is likely that some user input may land in the transition flow range where we interpolate `f` between the predictions of laminar and turbulent flow at that `Re` and `ϵ/d`, such that:
    - We match the predictions at `Re=2000` and `Re=4000`.
    - At intermediate `Re`, add some randomness by assuming `f` follows a normal distribution:
      - Mean: `μf = flam + (fCB - flam) * (Re - 2000) / 2000`
      - Standard deviation: `σf = 0.2 * μf`
"""
# endregion

# region imports
import hw5a as pta
import random as rnd
from matplotlib import pyplot as plt
import numpy as np
# endregion

# region constants
GRAVITY = 32.2  # ft/s^2 (acceleration due to gravity)
WATER_VISCOSITY = 1.08e-5  # ft^2/s (kinematic viscosity of water at 60°F)
# endregion

# region functions
def ffPoint(Re, rr):
    """
    This function calculates the friction factor (f) based on the Reynolds number (Re) and relative roughness (rr).
    - For Re > 4000: Uses the Colebrook equation (turbulent flow).
    - For Re < 2000: Uses the laminar flow equation (f = 64/Re).
    - For 2000 <= Re <= 4000: Uses a probabilistic approach with a normal distribution.
    :param Re: Reynolds number
    :param rr: Relative roughness (ε/D)
    :return: Friction factor (f)
    """
    if Re >= 4000:
        return pta.ff(Re, rr, CBEQN=True)  # Turbulent flow (Colebrook equation)
    if Re <= 2000:
        return 64 / Re  # Laminar flow
    # Transitional flow (2000 < Re < 4000)
    CBff = pta.ff(4000, rr, CBEQN=True)  # Friction factor at Re=4000 (Colebrook)
    Lamff = 64 / 2000  # Friction factor at Re=2000 (laminar)
    mean = Lamff + (CBff - Lamff) * (Re - 2000) / 2000  # Linear interpolation
    sig = 0.2 * mean  # Standard deviation
    return rnd.normalvariate(mean, sig)  # Random value from normal distribution

def calculate_head_loss_per_foot(f, D, V):
    """
    Calculates the head loss per foot (hf/L) using the Darcy-Weisbach equation.
    :param f: Friction factor
    :param D: Pipe diameter in feet
    :param V: Flow velocity in ft/s
    :return: Head loss per foot (hf/L) in ft/ft
    """
    return f * (V**2) / (2 * GRAVITY * D)
#Used ChatGPT to help solve: Your program should allow the user to re-specify the parameters
# and keep track of each f by just adding a new icon to the Moody diagram with each new set of parameters.
#ChatGPT helped create #def plot_all_points_on_moody
def plot_all_points_on_moody(history):
    """
    Plots all points on the Moody diagram with markers (triangle for transition flow, circle otherwise).
    :param history: List of dictionaries containing Re, f, and is_transition for each set of parameters
    """
    plt.figure()  # Create a new figure
    for entry in history:
        Re = entry['Re']
        f = entry['f']
        is_transition = entry['is_transition']
        marker = '^' if is_transition else 'o'  # Triangle for transition, circle otherwise
        plt.loglog(Re, f, marker, markersize=10, markeredgecolor='red', markerfacecolor='none', label=f'Re={Re:.2e}, f={f:.4f}')
    plt.legend()
    pta.plotMoody()  # Show the Moody diagram

def main():
    """
    Main function to execute the program.
    """
    history = []  # List to store history of user inputs and results
    while True:
        # Step 1: Get user input
        D_inches = float(input("Enter pipe diameter (inches): "))  # Pipe diameter in inches
        epsilon_micro_inches = float(input("Enter pipe roughness (micro-inches): "))  # Pipe roughness in micro-inches
        flow_rate_gpm = float(input("Enter flow rate (gallons/min): "))  # Flow rate in gallons/min

        # Step 2: Convert inputs to consistent units
        D_feet = D_inches / 12  # Convert diameter to feet
        epsilon_feet = epsilon_micro_inches * 1e-6 / 12  # Convert roughness to feet
        rr = epsilon_feet / D_feet  # Relative roughness (ε/D)
        flow_rate_cfs = flow_rate_gpm * 0.002228  # Convert flow rate to ft^3/s

        # Step 3: Calculate flow velocity and Reynolds number
        A = np.pi * (D_feet**2) / 4  # Cross-sectional area of the pipe (ft^2)
        V = flow_rate_cfs / A  # Flow velocity (ft/s)
        Re = V * D_feet / WATER_VISCOSITY  # Reynolds number

        # Step 4: Calculate friction factor
        f = ffPoint(Re, rr)
        is_transition = 2000 < Re < 4000  # Check if flow is in the transition region

        # Step 5: Calculate head loss per foot
        hf_L = calculate_head_loss_per_foot(f, D_feet, V)
        print(f"Head loss per foot (hf/L): {hf_L:.6f} ft/ft")

        # Step 6: Store the results in history
        history.append({'Re': Re, 'f': f, 'is_transition': is_transition})

        # Step 7: Plot all points on the Moody diagram
        plot_all_points_on_moody(history)

        # Step 8: Ask the user if they want to re-specify parameters
        repeat = input("Do you want to re-specify parameters? (yes/no): ").strip().lower()
        if repeat != 'yes':
            break
# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion