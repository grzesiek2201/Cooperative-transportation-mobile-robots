from mpopt import mp
import numpy as np
import matplotlib.pyplot as plt
# from weightedlpsum import wlps
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


def single_mp(x0, xf):
    Q = np.diag([10, 10, 10])          # don't turn too sharply
    R = np.diag([10, 100])               # keep inputs small
    P = np.diag([1000, 1000, 1000])
    
    u0 = np.array([1., 1.])
    uf = np.array([1., 1.])
    
    X = []
    U = []
    T = []

    ocp = mp.OCP(n_states=3, n_controls=2, n_phases=1, solver="scpgen")

    ocp.dynamics[0] = dynamics
    X0 = np.array(x0)
    Xf = np.array(xf)
    ocp.x00[0] = X0
    ocp.xf0[0] = Xf
    ocp.u00[0] = u0
    ocp.uf0[0] = uf
    ocp.lbu[0], ocp.ubu[0] = [0, -math.pi/32], [.5, math.pi/32]
    ocp.running_costs[0] = lambda x, u, t: quadratic_cost(x, u, t, x0=Xf, u0=uf, Q=Q, R=R) + 1
    # ocp.running_costs[i] = lambda x, u, t: (ca.power(u[0] - uf2[0], 2) + ca.power(u[1] - uf2[1], 2))
    ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0]-Xf[0], xf[1]-Xf[1], xf[2]-Xf[2]]
    ocp.terminal_costs[0] = lambda xf, tf, x0, t0: quadratic_cost(x=Xf, x0=xf, Q=P)  # was (x=x0, x0=xf, Q=P)  # doesn't seem to work (works but not always)

    mpo, post = mp.solve(ocp, n_segments=10, poly_orders=1, scheme="LGR", plot=False, solve_dict={"ipopt.max_iter": 1000})
    x, u, t, _ = post.get_data()
    X.append(x)
    U.append(u)
    T.append(t)

    # fig = plt.figure()
    # ax = fig.gca()
    # # plot robot's footprint
    # n_arr = int(len(x[:,0]) / 10)
    # postures = [(x[:,0][i], x[:,1][i], x[:,2][i]) for i in range(0, len(x[:,0]), n_arr)]
    # width, height = robot_size[0]*2, robot_size[1]*2
    # point_of_rotation = np.array([width/2, height/2])
    # for posture in postures:
    #     rec = RotatingRectangle((posture[0], posture[1]), width=width, height=height, 
    #                     rel_point_of_rot=point_of_rotation,
    #                     angle=posture[2]*180.0/np.pi, color='black', alpha=0.9,
    #                     fill=None)
    #     ax.add_patch(rec)
    # plt.show()

    return X


if __name__ == '__main__':

    # path = [(5, 5, 0), (6.0, 6.0, 1.5707963267948966), (6.0, 10.0, 1.5707963267948966), (6.0, 14.0, 1.5707963267948966), (6.0, 18.0, 1.5707963267948966), 
    #         (6.0, 20.0, 1.5707963267948966), (7.0, 21.0, 0.0), (11.0, 21.0, 0.0), (15.0, 21.0, 0.0), (19.0, 21.0, 0.0), (23.0, 21.0, 0.0), 
    #         (24.0, 20.0, 4.71238898038469), (24.0, 16.0, 4.71238898038469), (24.0, 14.0, 4.71238898038469), (25.0, 13.0, 0.0), (29.0, 13.0, 0.0), 
    #         (33.0, 13.0, 0.0), (33.0, 13.0, 0.7853981633974483), (34.0, 14.0, 0.7853981633974483), (35.0, 15.0, 0.7853981633974483), (36.0, 16.0, 0.7853981633974483), 
    #         (37.0, 17.0, 0.7853981633974483), (38.0, 18.0, 0.7853981633974483), (39.0, 19.0, 0.7853981633974483), (40.0, 20.0, 0.7853981633974483),
    #         (41.0, 21.0, 0.7853981633974483), (42.0, 22.0, 0.7853981633974483), (42.0, 22.0, 1.5707963267948966), (42.0, 24.0, 1.5707963267948966), 
    #         (42.0, 25.0, 1.5707963267948966), (42.0, 25.0, 2.356194490192345)]

    # path = [(5, 5, 0), (6.0, 6.0, 1.5707963267948966), (6.0, 10.0, 1.5707963267948966), (6.0, 14.0, 1.5707963267948966), (6.0, 18.0, 1.5707963267948966),
    #         (6.0, 20.0, 1.5707963267948966), (7.0, 21.0, 0.0), (11.0, 21.0, 0.0), (15.0, 21.0, 0.0), (19.0, 21.0, 0.0), ]


    path = [(42.0, 25.0, 1.5707963267948966), (42.0, 25.0, 2.356194490192345)]
    path = [(26, 6, math.pi*3/2), (25, 5, math.pi)]

    Q = np.diag([10, 10, 100])          # don't turn too sharply
    R = np.diag([1, 1])               # keep inputs small
    P = np.diag([1000, 1000, 1000])

    # 1 phase, many solvers

    # Plotting the result
    RESOLUTION = (200, 200)
    XY_RANGE = [[-10., 30.], [-10., 15.]]
    POINT_SIZE = 2
    X = np.linspace(XY_RANGE[0][0], XY_RANGE[0][1], RESOLUTION[0])
    Y = np.linspace(XY_RANGE[1][0], XY_RANGE[1][1], RESOLUTION[1])
    # s = S + R + E
    p = P
    xx, yy = np.meshgrid(X, Y)
    XY = np.stack((xx.ravel(), yy.ravel()), axis=1)
    robot_size = [.5, .2]

    plt.figure()
    ax = plt.gca()

    X = []
    U = []
    T = []

    ocp = mp.OCP(n_states=3, n_controls=2, n_phases=1, solver="scpgen")
    for i in range(len(path)-1):
        ocp.dynamics[0] = dynamics
        X0 = np.array(path[i])
        Xf = np.array(path[i+1])
        ocp.x00[0] = X0
        ocp.xf0[0] = Xf
        ocp.lbu[0], ocp.ubu[0] = [0, -math.pi/2], [1, math.pi/2]
        Xf = path[i+1]
        uf = np.array([1., 0.])
        ocp.running_costs[i] = lambda x, u, t: quadratic_cost(x, u, t, x0=Xf, u0=uf, Q=Q, R=R) + 1
        # ocp.running_costs[i] = lambda x, u, t: (ca.power(u[0] - uf2[0], 2) + ca.power(u[1] - uf2[1], 2))
        ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0]-Xf[0], xf[1]-Xf[1], xf[2]-Xf[2]]
        ocp.terminal_costs[0] = lambda xf, tf, x0, t0: quadratic_cost(x=Xf, x0=xf, Q=P)  # was (x=x0, x0=xf, Q=P)  # doesn't seem to work (works but not always)

        mpo, post = mp.solve(ocp, n_segments=10, poly_orders=1, scheme="LGR", plot=False, solve_dict={"ipopt.max_iter": 1000})
        x, u, t, _ = post.get_data()
        X.append(x)
        U.append(u)
        T.append(t)

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

    plt.show()
    # mp.plt.show()
