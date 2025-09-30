import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spInt
import ODE as myODE
import Integrators as myInt


k_ODE = 1  # default ODE spring constant
m_ODE = 2  # default ODE mass
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
    return (
        (z - R * np.cos(theta))
        * np.sin(theta)
        / (R**2 + z**2 - 2 * R * z * np.cos(theta)) ** 1.5
    )


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


def reference_integral(theta_max, R=R_int, z=z_int):
    """
    Exact value of the integral for a given theta_max, R, z
    """
    term1 = (R - z * np.cos(theta_max)) / (
        z**2 * (R**2 + z**2 - 2 * R * z * np.cos(theta_max)) ** 0.5
    )
    term2 = (R - z) / (z**2 * (R**2 + z**2 - 2 * R * z) ** 0.5)
    return term1 - term2


# Plot global truncation error for the integral
N_list = [1, 10, 20, 30, 40, 50, 70, 100, 120, 150, 200, 500]
results_riemann = []
results_trapezoid = []
results_simpson = []
theta_max_truncation = np.pi/2
result_exact = reference_integral(theta_max_truncation)
for N in N_list:
    result_riemann = myInt.Riemann(integrand, 0, theta_max_truncation, N)
    result_trapezoid = myInt.Trapezoidal(integrand, 0, theta_max_truncation, N)
    result_simpson = myInt.Simpson(integrand, 0, theta_max_truncation, N)
    
    results_riemann.append(np.abs(result_riemann-result_exact))
    results_trapezoid.append(np.abs(result_trapezoid-result_exact))
    results_simpson.append(np.abs(result_simpson-result_exact))

plt.plot(np.log(N_list), np.log(results_riemann), label="Riemann")
plt.plot(np.log(N_list), np.log(results_trapezoid), label="Trapezoid")
plt.plot(np.log(N_list), np.log(results_simpson), label="Simpson")
plt.legend()
plt.title("Global Truncation Error for Integration")
plt.xlabel("Number of slices log(N)")
plt.ylabel("log(Error)")
plt.show()
