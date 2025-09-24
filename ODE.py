def solve(integrator, x0, v0, xdot, vdot, dt, t0, t_max):
    """Use the given integrator to solve the differential equation
    given by the functions xdot(x,v,t) and vdot(x,v,t),
    with initial values x0, v0,
    up to time t_max.
    """

    x_list = [x0]
    v_list = [v0]
    t_list = [t0]

    x = x0
    v = v0

    for t in range(t0, t_max, dt):
        x_new, v_new = integrator(x, v, xdot, vdot, t, dt)
        x_list.append(x_new)
        v_list.append(v_new)
        t_list.append(t)

        x = x_new
        v = v_new

    return x_list, v_list, t_list


def Euler(x, v, xdot, vdot, t, dt):
    """Advance one timestep using the explicit form of Euler's method."""

    x_new = x + xdot(x, v, t) * dt
    v_new = v + vdot(x, v, t) * dt

    return x_new, v_new


def RK4(x, v, xdot, vdot, t, dt):
    """Advance one timestep using the 4th-order Runge-Kutta method.
    See Equations 8.33a-e of Mark Newman, Computational Physics.
    """

    k1x = dt * xdot(x, v, t)
    k1v = dt * vdot(x, v, t)
    k2x = dt * xdot(x + k1x / 2, v + k1v / 2, t + dt / 2)
    k2v = dt * vdot(x + k1x / 2, v + k1v / 2, t + dt / 2)
    k3x = dt * xdot(x + k2x / 2, v + k2v / 2, t + dt / 2)
    k3v = dt * vdot(x + k2x / 2, v + k2v / 2, t + dt / 2)
    k4x = dt * xdot(x + k3x, v + k3v, t + dt)
    k4v = dt * vdot(x + k3x, v + k3v, t + dt)

    x_new = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    v_new = v + (k1v + 2 * k2v + 2 * k3v + k4v) / 6

    return x_new, v_new
