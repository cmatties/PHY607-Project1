import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spInt
import ODE as myODE


k_ODE = 1  # default ODE spring constant
m_ODE = 2  # default ODE mass
F0_ODE = 0.1  # default ODE driving force amplitude
omega_ODE = 4  # default ODE driving frequency
gamma_ODE = 0.5  # default ODE damping coefficient


def reference_ODE(
    t, x0, v0, t0, k=k_ODE, m=m_ODE, F0=F0_ODE, omega=omega_ODE, gamma=gamma_ODE
):
    """
    Exact solution to the ODE given the specified parameters
    """
    omega_prime = (k / m - (gamma / 2) ** 2) ** 0.5
    phi = (omega * gamma) / (k / m - omega**2)
    particular_coefficient = (F0 / m) * (
        (k / m - omega**2) ** 2 + omega**2 * gamma**2
    ) ** -0.5
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
    initial_conditions_vector = np.array(
        [
            x0 - particular_coefficient * np.cos(omega * t0 - np.arctan(phi)),
            v0 + omega * particular_coefficient * np.sin(omega * t0 - np.arctan(phi)),
        ]
    )
    C1, C2 = np.linalg.solve(intial_conditions_matrix, initial_conditions_vector)
    return (
        C1 * np.exp(-gamma / 2 * t) * np.cos(omega_prime * t)
        + C2 * np.exp(-gamma / 2 * t) * np.sin(omega_prime * t)
        + particular_coefficient * np.cos(omega * t - np.arctan(phi))
    )


def xdot(x, v, t):
    return v


def vdot(x, v, t, k=k_ODE, m=m_ODE, F0=F0_ODE, omega=omega_ODE, gamma=gamma_ODE):
    return F0 / m * np.cos(omega * t) - (k / m) * x - gamma * v

def vector_wrapper(t,y, k=k_ODE, m=m_ODE, F0=F0_ODE, omega=omega_ODE, gamma=gamma_ODE):
    x = y[0]
    v = y[1]
    return(np.array([xdot(x,v,t), vdot(x,v,t, k=k, F0=F0, omega=omega, gamma=gamma)]))

# Compare solutions to scipy integrators and exact solution
dt = 0.001
x0_comparison = 0
v0_comparison = 1
tmax_comparison = 20
xlist_Euler, vlist_Euler, tlist_Euler = myODE.solve(
    myODE.Euler, x0_comparison, v0_comparison, xdot, vdot, dt, 0, tmax_comparison
)
xlist_RK4, vlist_RK4, tlist_RK4 = myODE.solve(
    myODE.RK4, x0_comparison, v0_comparison, xdot, vdot, dt, 0, tmax_comparison
)

xlist_exact = []
for t in tlist_Euler:
    xlist_exact.append(reference_ODE(t, x0_comparison, v0_comparison, 0))

scipy_solution = spInt.solve_ivp(vector_wrapper, (0,tmax_comparison), np.array([x0_comparison, v0_comparison]), max_step = dt)
tlist_scipy = np.array(scipy_solution.t)
xlist_scipy = np.array(scipy_solution.y[0,:])

xlist_Euler = np.array(xlist_Euler)
xlist_RK4 = np.array(xlist_RK4)
xlist_exact = np.array(xlist_exact)

plt.plot(tlist_Euler, xlist_Euler, label = "Euler")
plt.plot(tlist_RK4, xlist_RK4, label = "RK4")
plt.plot(tlist_scipy, xlist_scipy, label="Scipy RK4")
plt.plot(tlist_Euler, xlist_exact, label = "Exact", color="black", ls="--")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Position")
plt.title(r"Comparing ODE Integration Schemes, $F_0=0.1$")
plt.show()

plt.clf()

# Compare solutions to scipy integrators and exact solution for a different set of parameters

dt = 0.001
xlist_Euler, vlist_Euler, tlist_Euler = myODE.solve(
    myODE.Euler, x0_comparison, v0_comparison, xdot, lambda a,b,c:vdot(a,b,c,F0=1), dt, 0, tmax_comparison
)
xlist_RK4, vlist_RK4, tlist_RK4 = myODE.solve(
    myODE.RK4, x0_comparison, v0_comparison, xdot, lambda a,b,c:vdot(a,b,c,F0=1), dt, 0, tmax_comparison
)

xlist_exact = []
for t in tlist_Euler:
    xlist_exact.append(reference_ODE(t, x0_comparison, v0_comparison, 0, F0 = 1))

scipy_solution = spInt.solve_ivp(lambda a,b: vector_wrapper(a,b,F0=1), (0,tmax_comparison), np.array([x0_comparison, v0_comparison]), max_step = dt)
tlist_scipy = np.array(scipy_solution.t)
xlist_scipy = np.array(scipy_solution.y[0,:])

xlist_Euler = np.array(xlist_Euler)
xlist_RK4 = np.array(xlist_RK4)
xlist_exact = np.array(xlist_exact)

plt.plot(tlist_Euler, xlist_Euler, label = "Euler")
plt.plot(tlist_RK4, xlist_RK4, label = "RK4")
plt.plot(tlist_scipy, xlist_scipy, label="Scipy RK4")
plt.plot(tlist_Euler, xlist_exact, label = "Exact", color="black", ls="--")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Position")
plt.title(r"Comparing ODE Integration Schemes, $F_0=1$")
plt.show()

plt.clf()


# Plot global error
error_list_Euler = []
error_list_RK4 = []
dt_list = np.linspace(1e-4, 1, num=100)
x0_error = 0
v0_error = 1
tmax_error = 10
for dt in dt_list:
    xlist_Euler, vlist_Euler, tlist_Euler = myODE.solve(
        myODE.Euler, x0_error, v0_error, xdot, vdot, dt, 0, tmax_error
    )
    xlist_RK4, vlist_RK4, tlist_RK4 = myODE.solve(
        myODE.RK4, x0_error, v0_error, xdot, vdot, dt, 0, tmax_error
    )
    x_final_Euler = xlist_Euler[-1]
    x_final_RK4 = xlist_RK4[-1]
    x_final_exact = reference_ODE(tlist_Euler[-1], x0_error, v0_error, 0)

    error_list_Euler.append(np.abs(x_final_Euler - x_final_exact))
    error_list_RK4.append(np.abs(x_final_RK4 - x_final_exact))

plt.plot(dt_list, error_list_Euler, label="Euler")
plt.plot(dt_list, error_list_RK4, label="RK4")
plt.legend()
plt.title("Global Truncation Error", size=20)
plt.xlabel("Timestep", size=13)
plt.ylabel(r"$|x(t_{\mathrm{final}})-x_{\mathrm{exact}}(t_{\mathrm{final}})|$", size=13)
plt.show()
