import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def u0(x):
   
    if x >= 0 and x < (3.6 / 2):
        return (x * 4) / 3.6  
    elif x >= 1.8 and x <= 3.6:  
        return ((3.6 - x) * 4) / 3.6
    else:
        return 0.0

def advection_diffusion_backward_euler(Tfinal, Nx, Nt, u0):
   
    # parameters
    a = 0.1  # diffusion coefficient (m**2/s)
    b = 7.0  # flow velocity (m/s)
    L = 3.6  # domain length (m)
   
    # step sizes
    dx = L / Nx
    dt = Tfinal / Nt # time step
   
    # spatial grid
    mesh_points = np.linspace(0, L, Nx + 1)
   
    # initialise solution matrix
    u = np.zeros((Nt + 1, Nx + 1))
   
    # initial condition
    u[0, :] = u0(mesh_points)

    # backward Euler for diffusion and advection terms
    diff = a * dt / (dx * dx)  
    adv = b * dt / (2 * dx)  
   
    # sparse matrix for the linear system
    main_diag = 1 + 2 * diff * np.ones(Nx+1)
    upper_diag = (-diff - adv) * np.ones(Nx)
    lower_diag = (-diff + adv) * np.ones(Nx)
   
    # periodic boundary conditions
    A = sparse.diags(
        [main_diag, upper_diag, lower_diag, [-diff - adv], [-diff + adv]],
        [0, 1, -1, Nx, -Nx],
        format = 'csc'
    )
   
    # time stepping
    for n in range(Nt):
        # Solve the linear system
        u[n+1, :] = spsolve(A, u[n, :])
   
    return u

# vectorise u0 for array operations
u0 = np.vectorize(u0)

# set parameters
Nx = 8
Nt = 4
Tfinal = 1
x_upper = 3.6  

u = advection_diffusion_backward_euler(Tfinal, Nx, Nt, u0)

# animation
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
    fig, animate,
    frames=Nt+1,
    interval=100,
    init_func=lambda: None,
    blit=False)

plt.show()