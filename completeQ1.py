import numpy as np
import matplotlib.pyplot as plt


def f(x, t):
    T = 0.6
    t_switch_on = 0.05
    t_switch_off = 0.1

    # %T creates a repeatable time window that applies to every cycle
    repeating_pos_t = t % T

   
    if t_switch_on <= repeating_pos_t < t_switch_off:
        # when x is in the left quadrant of the triangular wave
        # g(x) = slope * (x - start)
        if 0.3 <= x <= 0.7:
            return 25 * (x - 0.3)
        # when x is in the right quadrant of the triangular wave
        # g(x) = slope * (end - x)
        elif 0.7 < x <= 1.1:
            return 25 * (1.1 - x)
        else:
            return 0.0
    else:
        return 0.0
   
def calculate_double_simpsons_integral(Tmax, f, Nx, Nt):

    if Nx % 2 != 0 or Nt % 2 != 0:
        raise RuntimeError("Nx and Nt must be even")
       
    # x limits, t limits
    a_x = 0.0
    b_x = 1.6
    a_t = 0.0
    b_t = Tmax
   
    # calculate width of each subinterval
    hx = (b_x - a_x) / Nx
    ht = (b_t - a_t) / Nt
   
    # evaluation points in x, t direction
    xvals = np.linspace(a_x, b_x, Nx + 1)
    tvals = np.linspace(a_t, b_t, Nt + 1)

    # set weights for Simpson's rule
    weights_x = np.ones(Nx + 1)
    for i in range(1, Nx):
        if i % 2 == 0:
            weights_x[i] = 2
        else:
            weights_x[i] = 4

   
    weights_t = np.ones(Nt + 1)
    for j in range(1, Nt):
        if j % 2 == 0:
            weights_t[j] = 2
        else:
            weights_t[j] = 4
   
   
    total = 0.0
   
    # loop over points in x, t direction
    for j in range(Nt + 1):
        for i in range(Nx + 1):
            x = xvals[i]
            t = tvals[j]
            total += weights_x[i] * weights_t[j] * f(x, t)
   
    return ((hx / 3) * (ht / 3)) * total


vectorized_f = np.vectorize(f)

my_integral_top_limit = 1.4  
x = np.linspace(0, my_integral_top_limit, 1001)

# Make a plot of the function
plt.plot(x, vectorized_f(x, 0.07))
plt.xlabel("x")
plt.ylabel("f(x,t)")
plt.title("f(x,t) at t=0.07")
plt.grid(True)
plt.show()


Tfinal = 0.5
Nx = 360
Nt = 600

print("Integral evaluates to ",
      calculate_double_simpsons_integral(Tfinal, f, Nx, Nt), "\n")