import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spInt
import ODE as myODE


k_ODE = 1  # default ODE spring constant
m_ODE = 2  # default ODE mass
F0_ODE = 1  # default ODE driving force amplitude
omega_ODE = 1  # default ODE driving frequency
gamma_ODE = 1  # default ODE damping coefficient

def reference_ODE(
    t, x0, v0, t0, k=k_ODE, m=m_ODE, F0=F0_ODE, omega=omega_ODE, gamma=gamma_ODE
):
    """
    Exact solution to the ODE given the specified parameters
    """
    omega_prime = (k / m - (gamma / 2) ** 2) ** 0.5
    phi = (omega * gamma) / (k / m - omega**2)
    intial_conditions_matrix = np.array(
        [
            [
                np.exp(-gamma / 2 * t0) * np.cos(omega_prime * t0),
                np.exp(-gamma / 2 * t0) * np.sin(omega_prime * t0),
            ],
            [
                np.exp(-gamma / 2 * t0)
                * (
                    -gamma / 2 * np.cos(omega_prime * t0)
                    - omega_prime * np.sin(omega_prime * t0)
                ),
                np.exp(-gamma / 2 * t0)
                * (
                    -gamma / 2 * np.sin(omega_prime * t0)
                    + omega_prime * np.cos(omega_prime * t0)
                ),
            ],
        ]
    )
    initial_conditions_vector = np.array([x0, v0])
    C1, C2 = np.linalg.solve(intial_conditions_matrix, initial_conditions_vector)
    return (
        C1 * np.exp(-gamma / 2 * t) * np.cos(omega_prime * t)
        + C2 * np.exp(-gamma / 2 * t) * np.sin(omega_prime * t)
        + (F0 / m)
        * ((k / m - omega**2) ** 2 + omega**2 * gamma**2) ** -0.5
        * np.cos(omega * t - np.arctan(phi))
    )
    

def xdot(x, v, t):
    return v


def vdot(x, v, t, k=k_ODE, m=m_ODE, F0=F0_ODE, omega=omega_ODE, gamma=gamma_ODE):
    return F0 / m * np.cos(omega * t) - (k / m) * x - gamma * v
