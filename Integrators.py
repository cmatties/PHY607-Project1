def Riemann(f, x_min, x_max, N):
    """Compute a left Riemann sum of f from x_min to x_max with N steps"""

    Delta_x = (x_max - x_min) / N
    Riemann_sum = 0
    for i in range(N):
        Riemann_sum += f(x_min + i * Delta_x) * Delta_x
    return Riemann_sum


def Trapezoidal(f, x_min, x_max, N):
    """Compute the integral of f from x_min to x_max using the trapezoidal rule with N steps"""

    Delta_x = (x_max - x_min) / N
    Trap_sum = f(x_min) / 2 + f(x_max) / 2
    for i in range(1, N):
        Trap_sum += f(x_min + i * Delta_x)
    Trap_sum *= Delta_x
    return Trap_sum


def Simpson(f, x_min, x_max, N):
    """Compute the integral of f from x_min to x_max using Simpson's rule with N steps"""

    Delta_x = (x_max - x_min) / N
    Trap_sum = f(x_min) + f(x_max)
    for i in range(1, N, 2):
        Trap_sum += 4 * f(x_min + i * Delta_x)
    for i in range(2, N, 2):
        Trap_sum += 2 * f(x_min + i * Delta_x)
    Trap_sum *= Delta_x / 3
    return Trap_sum
