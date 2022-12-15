from mpi4py import MPI
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from numpy import cos, sin

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
comm=MPI.COMM_WORLD

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 2.0  # length of pendulum 2 in m
M1 = 2.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -120.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])


# integrate your ODE using scipy.integrate.
    
y = integrate.odeint(derivs, state, t)

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])
    
x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                              interval=dt*1000, blit=True, init_func=init)
plt.show()


if rank == 0:
        print("proc 0 is starting")
        data1 = {'a': x1, 'b': y1}
        data2 = {'c': x2, 'd': y2}
        comm.ssend(data1, dest=1, tag=11)
        print("proc 0 sent message data1")
        comm.ssend(data2, dest=1, tag=12)
        print("proc 0 sent message data2")
        print("proc 0 is done")
        

elif rank == 1:
    
    print("proc 1 is starting")
    dataa = comm.recv(source=0, tag=11)
    datab = comm.recv(source=0, tag=12)
    print(dataa)
    print(datab)
    print("proc 1 is done")

@article{Momin_Mathematical_Modeling_of_2021,
author = {Momin, Abdul Aziz and Shende, Nikhil and Anamtatmakula, Abhijna and Ganguly, Emily and Gurbani, Ashwin and Joshi, Chaitanya A and Mahajan, Yogesh Y},
doi = {https://doi.org/10.48550/arXiv.2107.11737},
journal = {arXiv},
month = {7},
number = {11737},
pages = {1--8},
title = {{Mathematical Modeling of Heat Conduction}},
volume = {7},
year = {2021}
}
