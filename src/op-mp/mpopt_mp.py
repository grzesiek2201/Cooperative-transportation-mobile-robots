from mpopt import mp
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import aerosandbox.numpy as np  # to help with writing functions as numpy instead of casadi, not used currently
import pandas as pd
from pathlib import Path
import math
from rotrectangle import RotatingRectangle


def dynamics(x, u, t):  # kinematics?
    return [np.cos(x[2]) * u[0],
            np.sin(x[2]) * u[0],
            u[1]]


def sqrt_cost(x, u, t):
    return ca.sqrt(ca.power(ca.cos(x[2]) * u[0], 2) +            # xdot = cos(theta) v
                   ca.power(ca.sin(x[2]) * u[0], 2))             # ydot = sin(theta) v
    # return np.sqrt(x[0] * x[0] + x[1] * x[1])


def quadratic_cost(x, u=0, t=0, x0=0, u0=0, Q=None, R=None):
    """
    cost = (x - x0)^T Q (x - x0) + (u - u0)^T R (u - u0)

    Args:
        x (_type_): _description_
        u (int, optional): _description_. Defaults to 0.
        t (int, optional): _description_. Defaults to 0.
        x0 (int, optional): _description_. Defaults to 0.
        u0 (int, optional): _description_. Defaults to 0.
        Q (_type_, optional): _description_. Defaults to None.
        R (_type_, optional): _description_. Defaults to None.

    Returns:
        float: cost
    """
    if Q is None:
        return (u-u0).T @ R @ (u-u0)
    if R is None:
        return (x-x0).T @ Q @ (x-x0)
    return (x-x0).T @ Q @ (x-x0) + (u-u0).T @ R @ (u-u0)


def save_to_csv(x, u, t):
    # save results to .csv
    results = pd.DataFrame(np.hstack((x, u, t)))
    results.columns = ["x", "y", "0", "v", "w", "t"]
    path = list(Path(__file__).parent.parent.glob(f"op-mp/trajectory.csv"))[0]
    results.to_csv(path, index=False)


def solve(path, dynamics, P, Q, R, robot_size=[1.0, 1.0], plot=False):

    ocp = mp.OCP(n_states=3, n_controls=2, n_phases=1)#, solver="scpgen")

    robot_size = robot_size

    if plot:
        plt.figure()
        ax = plt.gca()

    u0 = np.array([1., 0.])
    uf = np.array([1., 0.])

    X = np.array(path[0])
    U = np.array([.0, .0])
    T = np.array([0])
    for i in range(len(path)-1):
        ocp.dynamics[0] = dynamics
        X0 = np.array(path[i])
        Xf = np.array(path[i+1])
        ocp.x00[0] = X0
        ocp.xf0[0] = Xf
        ocp.u00[0] = U[-1]
        ocp.uf0[0] = uf
        ocp.lbu[0], ocp.ubu[0] = [0, -math.pi/32], [.5, math.pi/32]
        # Xf = path[i+1]
        ocp.running_costs[0] = lambda x, u, t: quadratic_cost(x, u, t, x0=Xf, u0=uf, Q=Q, R=R) + 1
        # ocp.running_costs[0] = lambda x, u, t: 1/2*(ca.power(u[0] - uf[0], 2) + ca.power(u[1] - uf[1], 2))
        ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0]-Xf[0], xf[1]-Xf[1], xf[2]-Xf[2]]
        ocp.terminal_costs[0] = lambda xf, tf, x0, t0: quadratic_cost(x=Xf, x0=xf, Q=P)  # was (x=x0, x0=xf, Q=P)  # doesn't seem to work (works but not always)

        mpo, post = mp.solve(ocp, n_segments=50, poly_orders=1, scheme="LGR", plot=False, solve_dict={"ipopt.max_iter": 1000})
        x, u, t, _ = post.get_data()
        X = np.vstack((X, x[1:-1]))
        U = np.vstack((U, u[1:-1]))
        T = np.vstack((T, t[1:-1]+T[-1]))

        if plot:
            plt.plot(x[:,0], x[:,1], linewidth=1, c='r')
            plt.plot(X0[0], X0[1], 'ro', Xf[0], Xf[1], 'ro')

            # plot robot's footprint
            n_arr = int(len(x[:,0]) / 10)
            postures = [(x[:,0][i], x[:,1][i], x[:,2][i]) for i in range(0, len(x[:,0]), n_arr)]
            width, height = robot_size[0]*2, robot_size[1]*2
            point_of_rotation = np.array([width/2, height/2])
            for posture in postures:
                rec = RotatingRectangle((posture[0], posture[1]), width=width, height=height, 
                                rel_point_of_rot=point_of_rotation,
                                angle=posture[2]*180.0/np.pi, color='black', alpha=0.9,
                                fill=None)
                ax.add_patch(rec)

    if plot:
        plt.show()

    return X, U, T


def optimize(path, robot_size, plot=False):
    for i in range(len(path)):
        if path[i][2] >= math.pi:
            # path[i][2] -= 2*math.pi
            # path[i] = (path[i][0], path[i][1], path[i][2] - 2*math.pi)
            path[i] = (path[i][0], path[i][1], path[i][2] % (2*math.pi))
    Q = np.diag([10, 10, 10])          # don't turn too sharply
    R = np.diag([10, 100])               # keep inputs small
    P = np.diag([1000, 1000, 1000])

    X, U, T = solve(path, dynamics, P, Q, R, robot_size=robot_size, plot=plot)
    save_to_csv(X, U, T)


def main():

    Q = np.diag([0, 0, 0])          # don't turn too sharply
    R = np.diag([1, 10])               # keep inputs small
    P = np.diag([1000, 1000, 1000])

    # # many phases, one solver
    # ocp = mp.OCP(n_states=3, n_controls=2, n_phases=1, solver="scpgen")
    # for i in range(len(path)-1):
    #     ocp.dynamics[i] = dynamics
    #     X0 = np.array(path[i])
    #     Xf = np.array(path[i+1])
    #     ocp.x00[i] = X0
    #     ocp.xf0[i] = Xf
    #     ocp.lbu[i], ocp.ubu[i] = [0, -math.pi/2], [1, math.pi/2]
    #     Xf = path[i+1]
    #     uf = np.array([1., 0.])
    #     # ocp.running_costs[i] = lambda x, u, t: quadratic_cost(x, u, t, x0=Xf, u0=uf, Q=Q, R=R) + 1
    #     # ocp.running_costs[i] = lambda x, u, t: (ca.power(u[0] - uf2[0], 2) + ca.power(u[1] - uf2[1], 2))
    #     ocp.terminal_constraints[i] = lambda xf, tf, x0, t0: [xf[0]-Xf[0], xf[1]-Xf[1], xf[2]-Xf[2]]
    #     ocp.terminal_costs[i] = lambda xf, tf, x0, t0: quadratic_cost(x=Xf, x0=xf, Q=P)  # was (x=x0, x0=xf, Q=P)  # doesn't seem to work (works but not always)


    # 1 phase, many solvers

    # Plotting the result

    robot_size = [3, 1]

    X, U, T = solve(path, dynamics, P, Q, R, robot_size, plot=True)
    # X, U, T = solve(path, dynamics, P, Q, R)
    save_to_csv(X, U, T)


if __name__ == '__main__':
    main()
