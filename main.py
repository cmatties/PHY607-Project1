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

# Compare integrals to the exact solution and scipy
N_comparison = 200
theta_max_list = np.linspace(0, np.pi, num = 25)
my_riemann_list = []
my_trapezoid_list = []
my_simpson_list = []
exact_list = []
scipy_trapezoid_list = []
scipy_simpson_list = []
for theta_max in theta_max_list:
    my_riemann_list.append(myInt.Riemann(integrand, 0, theta_max, N_comparison))
    my_trapezoid_list.append(myInt.Trapezoidal(integrand, 0, theta_max, N_comparison))
    my_simpson_list.append(myInt.Simpson(integrand, 0, theta_max, N_comparison))
    
    theta_array = np.linspace(0, theta_max, num = N_comparison+1, endpoint = True)
    dtheta = theta_max/N_comparison
    
    scipy_simpson_list.append(spInt.simpson(integrand(theta_array), dx = dtheta))
    scipy_trapezoid_list.append(spInt.trapezoid(integrand(theta_array), dx = dtheta))
    
    exact_list.append(reference_integral(theta_max))

# Convert the lists of results to array to make plotting easier
my_riemann_array = np.array(my_riemann_list)
my_trapezoid_array = np.array(my_trapezoid_list)
my_simpson_array = np.array(my_simpson_list)
scipy_trapezoid_array = np.array(scipy_trapezoid_list)
scipy_simpson_array = np.array(scipy_simpson_list)
exact_array = np.array(exact_list)

fig, ax = plt.subplots(3,1)
fig.suptitle("Comparing Integrator Implementations and Exact Solutions")
ax[0].set_title("Riemann Sum")
ax[0].semilogy(theta_max_list, np.abs(my_riemann_array-exact_array)/exact_array, color="black", label="Exact")

ax[1].set_title("Trapezoidal Rule")
ax[1].semilogy(theta_max_list, np.abs(my_trapezoid_array-scipy_trapezoid_array)/scipy_trapezoid_array, label="SciPy")
ax[1].semilogy(theta_max_list, np.abs(my_trapezoid_array-exact_array)/exact_array, color="black", label="Exact")

ax[2].set_title("Simpson's Rule")
ax[2].semilogy(theta_max_list, np.abs(my_simpson_array-scipy_simpson_array)/scipy_simpson_array, label="SciPy")
ax[2].semilogy(theta_max_list, np.abs(my_simpson_array-exact_array)/exact_array, label="Exact", color="black")
ax[2].set_xlabel(r"$\theta_{\mathrm{max}}$")

ax[0].legend()
ax[1].legend()
ax[2].legend()
fig.tight_layout()
plt.show()

# Plot global truncation error for the integral
plt.clf()
N_list = [1, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 500]
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

plt.loglog(N_list, results_riemann, label="Riemann")
plt.loglog(N_list, results_trapezoid, label="Trapezoid")
plt.loglog(N_list, results_simpson, label="Simpson")
plt.legend()
plt.title("Global Truncation Error for Integration")
plt.xlabel("Number of slices N")
plt.ylabel("Error")
plt.show()
