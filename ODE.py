def solve_ODE(integrator, x0, v0, xdot, vdot, dt, t0, t_max):
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
