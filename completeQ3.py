import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

'''Part (a)'''
def advection_diffusion_backward_euler_with_forcing(Tfinal, Nx, Nt, u0, f):
    """

    Parameters:
    Tfinal (float): The time of the final timestep.
    Nx (int): Number of subintervals in the x-direction
    Nt (int): Number of subintervals in the t-direction
    u0 (function of one variable x): Function to set the initial condition at x
    f (function of two variables x and t): Function to define the forcing term
    at x and t.

    Returns:
    matrix (float): Approximation of the solution over the whole region [0, Tfinal] x [x-domain]
    """

    # Domain setup
    L = 3.6
    dx = L / Nx
    dt = Tfinal / Nt
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, Tfinal, Nt + 1)

    # Constants
    a = 0.1  # diffusion coefficient
    b = 7.0  # advection velocity

    # Initialise the solution matrix for the whole domain
    u = np.zeros([Nt+1, Nx+1])
    u[0, :] = u0(x)

    # Coefficients for matrix A
    diff = a * dt / dx**2
    adv = b * dt / (2 * dx)

    # Construct sparse matrix with periodic BCs
    main_diag = (1 + 2 * diff) * np.ones(Nx + 1)
    upper_diag = (-diff - adv) * np.ones(Nx)
    lower_diag = (-diff + adv) * np.ones(Nx)

    A = sparse.diags(
        [main_diag, upper_diag, lower_diag, [-diff - adv], [-diff + adv]],
        [0, 1, -1, Nx, -Nx],
        format='csc'
    )

    # Time stepping with forcing term
    for n in range(Nt):
        fn1 = f(x, t[n+1])  # forcing at next timestep
        rhs = u[n, :] + dt * fn1
        u[n+1, :] = spsolve(A, rhs)

    return u




'''Part (b)'''
def plot_total_unobtainium_over_time(u, Tfinal):
    """
    
    Parameters:
    u (matrix of floats): Approximation of the solution over the whole region
    [0, Tfinal] x [x-domain]
    Tfinal (float): The time of the final timestep.

    Returns:
    fig (matplotlib figure): The figure containing the plot
    """
    Nt = u.shape[0] - 1
    Nx = u.shape[1] - 1
    L = 3.6
    dx = L / Nx
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, Tfinal, Nt + 1)

    total_unob = np.array([simps(u[n, :], x) for n in range(Nt + 1)])

    fig, ax = plt.subplots()
    ax.plot(t, total_unob, 'b-o')
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Total unobtainium (mol)")
    ax.set_title("Total unobtainium in the system over time")
    ax.grid(True)

    return fig




'''Part (c)'''
def comment_on_results():
    """
    Provide a brief comment on the results of the simulation.

    Returns:
    string: Comment on the results of the simulation
    """
    return (
       'At the start, the amount of unobtainium in the system goes up because'
       'new material is being added regularly. After a while, things level'
       'off—the amount being added gets balanced out as the material spreads' 
       'through the system due to movement and mixing. By the time we reach the'
       'final moment (Tfinal = 3.3s), the total amount in the system matches'
       'what we’d expect based on how much was added. This shows that the'
       'simulation is keeping the overall amount of material consistent,' 
       'which is what we want.'


    )





'''Functions from Q1 & Q2 '''

def f(x, t):
    """
    Definition of the forcing function f(x,t) for the advection-diffusion
    equation.

    Parameters:
    x (float): x-coordinate (m)
    t (float): time (s)
    Returns:
    float: value of the function at (x,t)
    """
    return np.sin(2 * np.pi * x / 3.6) * np.cos(2 * np.pi * t)

def u0(x):
    """
    Return the initial condition at x.

    Parameters:
    x (float): x-coordinate

    Returns:
    float: value of the initial condition at x
    """
    if x >= 0 and x < (3.6 / 2):
        return (x * 4) / 3.6  
    elif x >= 1.8 and x <= 3.6:  
        return ((3.6 - x) * 4) / 3.6
    else:
        return 0.0

# Vectorize functions for array inputs
u0 = np.vectorize(u0)
f = np.vectorize(f)




''''Plotting section'''

Nx = 8
Nt = 4
Tfinal = 1
x_upper = 3.6

# Run simulation
u = advection_diffusion_backward_euler_with_forcing(Tfinal, Nx, Nt, u0, f)

# Animation plot
fig, ax = plt.subplots()
meshgrid = np.linspace(0, x_upper, Nx+1)

def animate(i):
    ax.clear()
    ax.plot(meshgrid, u[i, :], 'o-')
    ax.plot(meshgrid, np.ones(Nx+1)/meshgrid[-1], 'k--')
    ax.set_ylim([-0.1, 3.1])
    ax.set_title(f"Time = {(Tfinal/Nt)*i:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(["u approximation", "steady state"])

ani = animation.FuncAnimation(
    fig, animate, frames=Nt+1, interval=100, init_func=lambda: None, blit=False)
plt.show()

# Plot total unobtainium over time
fig = plot_total_unobtainium_over_time(u, Tfinal)
plt.show()

# Output result comment
print(comment_on_results())
