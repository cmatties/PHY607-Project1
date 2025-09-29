import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spInt
import ODE as myODE
import Integrators as myInt


k_ODE = 1  # default ODE spring constant
m_ODE = 1  # default ODE mass
F0_ODE = 1  # default ODE driving force amplitude
omega_ODE = 1  # default ODE driving frequency
gamma_ODE = 1  # default ODE damping coefficient

R_int = 1  # default integral sphere radius
z_int = 2 * R_int  # default integral z coordinate
m_int = 1  # Default integral mass


def xdot(x, v, t):
    return v


def vdot(x, v, t, k=k_ODE, m=m_ODE, F0=F0_ODE, omega=omega_ODE, gamma=gamma_ODE):
    return F0 / m * np.cos(omega * t) - (k / m) * x - gamma * v


def integrand(theta, R=R_int, z=z_int):
    return (R**2 + z**2 - 2 * R * z * np.cos(theta)) ** (-1)
