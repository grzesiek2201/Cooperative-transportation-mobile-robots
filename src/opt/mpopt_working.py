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


def quadratic_and_obstacle(x, u, t, x0=0, u0=0, Q=None, R=None, sigmas=None, coords=None):
    # not working currently due to not being able to determine if wlps_ca() > 0
    cost = quadratic_cost(x=x0, x0=x, Q=Q)
    if sigmas is None or coords is None:
        return cost
    obstacle_cost = [1 for s, [p, h, k, a] in zip(sigmas, coords) if -wlps_ca(x, s, p, h, k, a) + 1 > 0]
    if sum(obstacle_cost) > 0:
        return float('inf')
    return cost


def wlps(xy, s, p):
    norm = np.power(np.sum(np.power(np.abs(xy)/s, p)), 1/p)
    return norm


def wlpn(x, s, p, h, k, a):
    """
    Weighted Lp norm for R2 with translation and rotation.

    Args:
        x (list[float]): values to compute
        s (list[float]): sigma vector (vector of weights) 
        p (float): p parameter of the Lp norm
        h (float): horizontal shift, positive value shifts to the right
        k (float): vertical shift, positive value shifts up
        a (float): rotation angle in radians, positive shifts according to coordinate system handedness

    Returns:
        float: value of the norm
    """
    return np.power(np.power(np.abs((x[0] - h)*np.cos(a)+(x[1]-k)*np.sin(a)) / s[0], p) +\
                    np.power(np.abs((x[0] - h)*np.sin(a)-(x[1]-k)*np.cos(a)) / s[1], p), 1/p)


def rotate_point(rectangle, angle, point):
    """Rotate rectangle around point

    Args:
        rectangle (np.ndarray[list[float, float]]): Numpy array with coordinates of the 4 rectangle's corners
        angle (float): Angle the rectangle is to be rotated by in radians
        point (list[float, float]): Coordinates of point around which the rectangle is to be rotated

    Returns:
        np.ndarray[list[float, float]]: Numpy array with rotated coordinates of the 4 rectange's corners
    """
    return [(point[0] + (corner[0] - point[0]) * np.cos(angle) - (corner[1] - point[1]) * np.sin(angle),
             point[1] + (corner[0] - point[0]) * np.sin(angle) + (corner[1] - point[1]) * np.cos(angle),) 
            for corner in rectangle]


def rrcollision(robot, obstacle, i):
    # robot and obstacle are in params form [x, y, angle, s1, s2] [x, y, angle, s1, s2, p]
    # obstacle in robot's frame
    if i < 4:
        obstacle_corners = vert_from_params(obstacle)
        obstacle_r = to_rframe(rectangle=obstacle_corners, robot=robot, deg=False)
        collision = wlps_ca_simple(obstacle_r[i][0], obstacle_r[i][1], robot[3:5], robot[5])
    # check for collision type A
    # collisions_A = ([-wlps_ca_simple(corner[0], corner[1], robot[3:], p)+1 for corner in obstacle_r])
    # collisions_A = ([wlps(co)])
    # if any(collisions_A <= 1):
    #     print("Collision type A")

    # robot in obstacle frame
    else:
        robot_corners = vert_from_params(robot)
        robot_o = to_oframe(robot=robot_corners, rectangle=obstacle, deg=False)
        collision = wlps_ca_simple(robot_o[i-4][0], robot_o[i-4][1], obstacle[3:5], obstacle[5])
    # check for collision type B
    # collisions_B = ([-wlps_ca_simple(corner[0], corner[1], obstacle[3:], p)+1 for corner in robot_o])
    # if any(collisions_B <= 1):
    #     print("Collision type B")
    # return collisions_A + collisions_B

    return collision


#### Casadi functions

def vert_from_params(params):
    # get rectangle vertices from its parameters
    center = ca.SX.sym("center", 2)
    center[0], center[1] = params[0], params[1]
    angle = params[2]
    dimensions = params[3], params[4]

    # create the (normalized) perpendicular vectors
    v1 = ca.SX.sym("v1", 2)
    v2 = ca.SX.sym("v2", 2)
    v1[0], v1[1] = ca.cos(angle), ca.sin(angle)
    v2[0], v2[1] = -v1[1], v1[0]

    # scale them appropriately by the dimensions
    v1 *= dimensions[0]
    v2 *= dimensions[1]

    # return the corners by moving the center of the rectangle by the vectors
    ret = ca.SX.sym("corners", 4, 2)
    ret[0, :] = center + v1 + v2
    ret[1, :] = center - v1 + v2
    ret[2, :] = center - v1 - v2
    ret[3, :] = center + v1 - v2
    return ret

def to_rframe(rectangle, robot, deg=False):
    # transform rectangle from global to robot frame
    # deg: if robot's angle in degrees instead of radians
    translated = translate_corners(rectangle=rectangle, xy=-robot)
    if deg:
        angle = -robot[2] * ca.pi / 180
    else:
        angle = -robot[2]
    transformed = rotate_origin(rectangle=translated, angle=angle)
    return transformed

def to_oframe(robot, rectangle, deg=False):
    # transform robot from global to obstacle frame
    translated = translate_corners(rectangle=robot, xy=-rectangle)
    if deg:
        angle = -rectangle[2] * ca.pi / 180
    else:
        angle = -rectangle[2]
    transformed = rotate_origin(rectangle=translated, angle=angle)
    return transformed

def translate_corners(rectangle, xy):
    return [(corner[0] + xy[0], corner[1] + xy[1]) for corner in ca.vertsplit(rectangle, 1)]

def rotate_origin(rectangle, angle, point=False):
    # rotate rectangle around (0, 0)
    if not point:
        return [(ca.cos(angle) * corner[0] - ca.sin(angle) * corner[1], ca.sin(angle) * corner[0] + ca.cos(angle) * corner[1]) for corner in rectangle]

    return (ca.cos(angle) * rectangle[0] - ca.sin(angle) * rectangle[1], ca.sin(angle) * rectangle[0] + ca.cos(angle) * rectangle[1])


def save_to_csv(x, u, t, filename="trajectory.csv"):
    # save results to .csv
    results = pd.DataFrame(np.hstack((x, u, t)))
    results.columns = ["x", "y", "0", "v", "w", "t"]
    path = list(Path(__file__).parent.parent.glob(f"testing/{filename}"))[0]
    results.to_csv(path, index=False)


def plot_results(x, sigmas, coords, X0, Xf, robot_size, p):
    # Plotting the result
    RESOLUTION = (200, 200)
    XY_RANGE = [[-10., 30.], [-10., 15.]]
    POINT_SIZE = 2

    X = np.linspace(XY_RANGE[0][0], XY_RANGE[0][1], RESOLUTION[0])
    Y = np.linspace(XY_RANGE[1][0], XY_RANGE[1][1], RESOLUTION[1])
    # s = S + R + E
    xx, yy = np.meshgrid(X, Y)
    XY = np.stack((xx.ravel(), yy.ravel()), axis=1)
    plt.figure()

    result = np.array([]).reshape(-1, 2)
    for s, [p, h, k, a] in zip(sigmas,coords):
        temp = np.array([xy for xy in XY if -wlpn(xy, s, p, h, k, a) + 1 > 0]).reshape(-1, 2)
        result = np.vstack((result, temp))
    plt.scatter(result[:, 0], result[:, 1])

    # plt.subplot(3, 1, 1)
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linewidth=0.5, alpha=.5)
    ax.xaxis.grid(color='gray', linewidth=0.5, alpha=.5)
    plt.plot(x[:,0], x[:,1], linewidth=1, c='r')
    plt.plot(X0[0], X0[1], 'ro', Xf[0], Xf[1], 'ro')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Planned path")
    ax.set_aspect('equal')
    ax = plt.gca()

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

    # circle1 = plt.Circle((circle[0], circle[1]), circle[2], color='r')
    # ax.add_patch(circle1)
    # ax.set_aspect('equal', adjustable='box')
    plt.show()
    mp.plt.show()


def define_obstacles(mode='circ-rect'):
    # for the circ-rect example
    if mode == 'circ-rect':
        sigmas = [[8, 1], [8, 1], [2, 6], [.5, 2], [8, 1], [8, 1], [2, 6], [.5, 2], [8, 1], [8, 1], [2, 6], [.5, 2]]
        coords = [[20, 10, 3, 0.3], [20, 10, 0, 0.3], [20, 22.5, 3, 0], [20, 24, 0, 0],
                [20, 10, 3, 0.3], [20, 10, 0, 0.3], [20, 22.5, 3, 0], [20, 24, 0, 0],
                [20, 10, 3, 0.3], [20, 10, 0, 0.3], [20, 22.5, 3, 0], [20, 24, 0, 0]]  # [p - degree, h - horizontal transition, k - vertical transition, a - angle]

    # for the rect-rect example;
    if mode == 'rect-rect':
        # X0 = np.array([-5., -2., -math.pi/4]) Xf = np.array([6., 4, -math.pi/4])
        # ocp.lbu[0], ocp.ubu[0] = [-5, -math.pi/2], [5, math.pi/2]
        # original, doesn't work; kinda works with bounded time and robot_size [.9, .1]
        # sigmas = [[5, 1], [5, 1]]
        # coords = [[10, 2, 1-.6, math.pi/3], [10, 0, 2.5, math.pi/3]]
        # robot_size = [.9, 1.3]
        # working with robot_size = [.9, 1.5] but not with [.9, .5]
        sigmas = [[5, 1], [5, 1]]
        coords = [[20, 0, -.6, math.pi/3], [20, 0, 10.5, math.pi/3]]
        robot_size = [.9, 1.5]

    # for the circ-rect example
    if mode == 'circ-rect':
        # X0 = np.array([-3.11, -.11, -math.pi/4]) Xf = np.array([3.52, -.22, -math.pi/4])
        # tf = 3.667 * math.pi
        # ocp.lbu[0], ocp.ubu[0] = [-2*math.pi, -math.pi/2], [2*math.pi, math.pi/2]
        sigmas = [[1, 1], [1, 1]]
        coords = [[2, 2, -1.6, 0], [2, -1, 1.5, 0]]
        robot_size = [2, 1]

    # for demonstration purposes
    if mode == 'dem':
        # X0 = np.array([0, 0, 0]); u0=np.array([0, 0])
        # Xf = np.array([10, 2, math.pi/2]); uf = np.array([0., 0.])
        # ocp.lbu[0], ocp.ubu[0] = [-2*math.pi, -math.pi/2], [2*math.pi, math.pi/2]
        sigmas = [[1, 10], [1 ,2]]
        coords = [[15, 5, 2, 0], [15, 5, 0, 0]]
        # r = 0
        robot_size = [.5, .2]
        # sigmas = []
        # coords = []

    return sigmas, coords


x = ca.SX.sym('x', 3)
u = ca.SX.sym('u', 2)
s = ca.SX.sym('s', 2)
p = ca.SX.sym('p', 1)
h = ca.SX.sym('h', 1)
k = ca.SX.sym('k', 1)
a = ca.SX.sym('a', 1)
# wlps_ca = ca.Function('wlps', [x, u, s, p], [ca.power(ca.cumsum(ca.power((ca.fabs(ca.vertsplit(x, [0, 2, 3])[0]) / s), p)), 1/p)]) 
wlps_ca = ca.Function('wlps', [x, s, p, h, k, a], [ca.power(ca.power(ca.fabs((x[0] - h) * ca.cos(a) + (x[1] - k) * ca.sin(a)) / s[0], p) +\
                                                ca.power(ca.fabs((x[0] - h) * ca.sin(a) - (x[1] - k) * ca.cos(a)) / s[1], p), 1/p)])
xs = ca.SX.sym('xs', 1)
ys = ca.SX.sym('ys', 1)
wlps_ca_simple = ca.Function('wlps_simple', [xs, ys, s, p], [ca.power(ca.power(ca.fabs(xs) / s[0], p) +\
                                                                    ca.power(ca.fabs(ys) / s[1], p), 1/p)])
# for rect robot - circ obstacle
# xs, ys - robot position, s - robot sigma (halfwidths), x - obstacle position, p - norm rank 
wlps_ca_c = ca.Function('wlps_c', [xs, ys, s, x, p], [ca.power( ca.power(ca.fabs(x[0] - xs) / s[0], p) +\
                                                                ca.power(ca.fabs(x[1] - ys) / s[1], p), 1/p)])


def main():

    r = 1
    X0 = np.array([5, 5, 0]); u0 = np.array([0., 0.])
    Xf = np.array([30, 15, 0]); uf = np.array([0., 0.])
    tf = 25
    circle = (5, 0, 3)
    robot_size = [3, 1]

    # Define OCP
    ocp = mp.OCP(n_states=3, n_controls=2, n_phases=1, solver="scpgen")
    ocp.dynamics[0] = dynamics
    ocp.x00[0] = X0
    ocp.xf0[0][0] = Xf[0]; ocp.xf0[0][1] = Xf[1]; #ocp.xf0[0][2] = Xf[2]
    # ocp.u00[0] = u0
    # ocp.uf0[0] = uf
    ocp.lbu[0], ocp.ubu[0] = [-2, -math.pi/2], [2, math.pi/2]  # if the input constraints are too restrictive the solution is often not found
    ocp.lbtf[0] = tf
    ocp.ubtf[0] = tf*2
    # ocp.scale_t = 1 / (2*tf)

    sigmas, coords = define_obstacles(mode="dem")

    sigmas = [[5, .5], [.5 ,7], [10, .5]]
    coords = [[15, 15, 15, 0], [15, 20, 8, 0], [15, 10, 1, 0]]

    # also have to add offset to the obstacles +r +e ******************
    sigmas = [np.array(sigma) for sigma in sigmas]
    ocp.path_constraints[0] = lambda x, u, t: [-wlps_ca(x, s+r, p, h, k, a) + 1 for s, [p, h, k, a] in zip(sigmas, coords)]
    # ocp.path_constraints[0] = lambda x, u, t: [-rrcollision(np.array([x[0], x[1], x[2], robot_size[0], robot_size[1], 8]), np.array([h, k, a, s[0], s[1], p]), i) + 1
    #                                            for s, [p, h, k, a] in zip(sigmas, coords) for i in range(8)]
    # rect robot - circ obstacle
    # ocp.path_constraints[0] = lambda x, u, t: [-wlps_ca_simple(rotate_origin([h - x[0], k - x[1]], x[2], point=True)[0], 
    #                                                            rotate_origin([h - x[0], k - x[1]], x[2], point=True)[1], robot_size + s, 20) + 1 
    #                                            for s, [p, h, k, a] in zip(sigmas, coords)]

    # ocp.running_costs[0] = lambda x, u, t: 1
    # ocp.running_costs[0] = sqrt_cost
    # ocp.running_costs[0] = lambda x, u, t: 0.5 * (u[0] * u[0] + u[1] * u[1])
    Q = np.diag([0, 0, 1])          # don't turn too sharply
    R = np.diag([1, 1])               # keep inputs small
    P = np.diag([1000, 1000, 1000])
    ocp.running_costs[0] = lambda x, u, t: quadratic_cost(x, u, t, x0=Xf, u0=uf, Q=Q, R=R) + 1
    # ocp.running_costs[0] = lambda x, u, t: quadratic_and_obstacle(x, u, t, x0=Xf, u0=uf, Q=Q, R=R, sigmas=sigmas, coords=coords)
    # ocp.running_costs[0] = lambda x, u, t: (1/2 * (u[0] * u[0] + u[1] * u[1]))
    # ocp.running_costs[0] = lambda x, u, t: ca.power(u[0] - uf[0], 2) + ca.power(u[1] - uf[1], 2)  # not usable
    ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0]-Xf[0], xf[1]-Xf[1], xf[2]-Xf[2]]
    ocp.terminal_costs[0] = lambda xf, tf, x0, t0: quadratic_cost(x=Xf, x0=xf, Q=P)  # was (x=x0, x0=xf, Q=P)  # doesn't seem to work (works but not always)
    # ocp.terminal_costs[0] = lambda xf, tf, x0, t0: tf  # the final cost is the final time
    # ocp.terminal_costs[0] = lambda xf, tf, x0, t0: 1* (ca.power(Xf[0] - xf[0], 2) + ca.power(Xf[1] - xf[1], 2) + ca.power(Xf[2] - xf[2], 2))  # works well with simple example


    # Create optimizer(mpo), solve and post process(post) the solution
    mpo, post = mp.solve(ocp, n_segments=170, poly_orders=5, scheme="LGR", plot=True, solve_dict={"ipopt.max_iter": 1000})
    # mpo, post = mp.solve(ocp, n_segments=10, poly_orders=10, scheme="CGL", plot=True)
    # mpo, post = mp.solve(ocp, n_segments=15, poly_orders=5, scheme="LGL", plot=True)
    x, u, t, _ = post.get_data()

    save_to_csv(x, u, t, filename="trajectory.csv")
    plot_results(x, sigmas, coords, X0, Xf, robot_size, p=15)



if __name__ == '__main__':
    main()
