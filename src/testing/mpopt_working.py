from mpopt import mp
import numpy as np
import matplotlib.pyplot as plt
# from weightedlpsum import wlps
import casadi as ca
import aerosandbox.numpy as np  # to help with writing functions as numpy instead of casadi, not used currently
import pandas as pd


def dynamics(x, u, t):  # kinematics?
    return [np.cos(x[2]) * u[0],
            np.sin(x[2]) * u[0],
            u[1]]


def sqrt_cost(x, u, t):
    return np.sqrt(np.power(np.cos(x[2]) * u[0], 2) +            # xdot = cos(theta) v
                   np.power(np.sin(x[2]) * u[0], 2))             # ydot = sin(theta) v
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


def wlps(x, u, s, p, n=2):
    # XY = np.vstack(list(zip(x,y)))
    xy = np.stack((x[0], x[1]))
    norm = np.power(np.sum(np.power(np.abs(xy)/s, p)), 1/p)
    # if norm < 1:
        # print(f"{x=}")
    return norm


def wlpn(x, s, p, h, k, a):
    """
    Weighted Lp norm for R2 with translation and rotation

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


def rotate_origin(rectangle, angle):
    """Rotate rectangle around (0, 0)

    Args:
        rectangle (np.ndarray[list[float, float]]): Numpy array with coordinates of the 4 rectangle's corners
        angle (float): Angle the rectangle is to be rotated by in radians

    Returns:
        np.ndarray[list[float, float]]: Numpy array with rotated coordinates of the 4 rectange's corners
    """
    # rotate rectangle around (0, 0)
    return np.array([(np.cos(angle) * corner[0] - np.sin(angle) * corner[1], np.sin(angle) * corner[0] + np.cos(angle) * corner[1]) for corner in rectangle])


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


def translate_corners(rectangle, xy):
    return [(corner[0] + xy[0], corner[1] + xy[1]) for corner in rectangle]


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

X0 = np.array([0., 0., 0.5]); u0 = np.array([0., 0.])
Xf = np.array([32., 2, 3]); uf = np.array([0., 0.])
tf = 30.0
circle = (5, 0, 3)

# Define OCP
ocp = mp.OCP(n_states=3, n_controls=2, n_phases=1)
ocp.dynamics[0] = dynamics
ocp.x00[0] = X0
ocp.xf0[0][0] = Xf[0]; ocp.xf0[0][1] = Xf[1]; ocp.xf0[0][2] = Xf[2]
ocp.u00[0] = u0
ocp.uf0[0] = uf
ocp.lbu[0], ocp.ubu[0] = [0, -1], [1, 1]  # if the input constraints are too restrictive the solution is often not found
ocp.lbtf[0] = tf
ocp.ubtf[0] = tf*2
# ocp.scale_t = 1 / tf

# ocp.path_constraints[0] = lambda x, u, t: [-(x[0] - circle[0])**2 - (x[1]-circle[1])**2 + circle[2]**2]
# the lambda function must return array of values that represent constraints, so iterate through all the obstacles and their transformations to robot frame
sigmas = [[8, 1], [8, 1], [2, 6], [.5, 2], [8, 1], [8, 1], [2, 6], [.5, 2], [8, 1], [8, 1], [2, 6], [.5, 2]]
coords = [[20, 10, 3, 0.3], [20, 10, 0, 0.3], [20, 22.5, 3, 0], [20, 24, 0, 0],
          [20, 10, 3, 0.3], [20, 10, 0, 0.3], [20, 22.5, 3, 0], [20, 24, 0, 0],
          [20, 10, 3, 0.3], [20, 10, 0, 0.3], [20, 22.5, 3, 0], [20, 24, 0, 0]]  # [p - degree, h - horizontal transition, k - vertical transition, a - angle]
# also have to add offset to the obstacles +r +e ******************
sigmas = [np.array(sigma) for sigma in sigmas]
r = 0.35
ocp.path_constraints[0] = lambda x, u, t: [-wlps_ca(x, s+r, p, h, k, a) + 1 for s, [p, h, k, a] in zip(sigmas, coords)]
# ocp.path_constraints[0] = lambda x, u, t: [-wlpn(x, [1.5, 1.5], 10, 3, 0, 1) + 1]
# ocp.path_constraints[0] = lambda x, u, t: [-x[0]+13 * x[0]-17 * -x[1]-2 * x[1]-2]  # doesn't work and I don't know if it should?

# ocp.running_costs[0] = sqrt_cost
# ocp.running_costs[0] = lambda x, u, t: 0.5 * (x[0] * x[0] + u[0] * u[0])
Q = np.diag([0, 0, 0.01])          # don't turn too sharply
R = np.diag([1, .1])               # keep inputs small
P = np.diag([1000, 1000, 1000])
ocp.running_costs[0] = lambda x, u, t: quadratic_cost(x, u, t, x0=Xf, u0=uf, Q=Q, R=R)
# ocp.running_costs[0] = lambda x, u, t: quadratic_and_obstacle(x, u, t, x0=Xf, u0=uf, Q=Q, R=R, sigmas=sigmas, coords=coords)
# ocp.running_costs[0] = lambda x, u, t: (u[0] * u[0] + u[1] * u[1])
ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0]-Xf[0], xf[1]-Xf[1], xf[2]-Xf[2]]
# ocp.terminal_costs[0] = lambda xf, tf, x0, t0: xf[0] - x0[0] + xf[1] - x0[1]
ocp.terminal_costs[0] = lambda xf, tf, x0, t0: quadratic_cost(x=x0, x0=xf, Q=P)
# ocp.scale_x = [
#     1 / 10.0,
#     1 / 10.0,
#     1 / 10.0
# ]

# Create optimizer(mpo), solve and post process(post) the solution
mpo, post = mp.solve(ocp, n_segments=100, poly_orders=1, scheme="LGR", plot=True, solve_dict={"ipopt.max_iter": 1000})
# mpo, post = mp.solve(ocp, n_segments=10, poly_orders=10, scheme="CGL", plot=True)
# mpo, post = mp.solve(ocp, n_segments=5, poly_orders=5, scheme="LGL", plot=True)
x, u, t, _ = post.get_data()

# save results to .csv
results = pd.DataFrame(np.hstack((x, u, t)))
results.columns = ["x", "y", "0", "v", "w", "t"]
results.to_csv("D:\\Projects\\cooperative_transportation\\src\\testing\\trajectory.csv", index=False)

# Plotting the result
RESOLUTION = (200, 200)
XY_RANGE = [[-10., 30.], [-10., 15.]]
S = np.array([2, 1])
P = 20
POINT_SIZE = 2
R = .5
E = 0.01

X = np.linspace(XY_RANGE[0][0], XY_RANGE[0][1], RESOLUTION[0])
Y = np.linspace(XY_RANGE[1][0], XY_RANGE[1][1], RESOLUTION[1])
# s = S + R + E
p = P
xx, yy = np.meshgrid(X, Y)
XY = np.stack((xx.ravel(), yy.ravel()), axis=1)
plt.figure()

result = np.array([]).reshape(-1, 2)
for s, [p, h, k, a] in zip(sigmas,coords):
    temp = np.array([xy for xy in XY if -wlpn(xy, s, p, h, k, a) + 1 > 0]).reshape(-1, 2)
    result = np.vstack((result, temp))
plt.scatter(result[:, 0], result[:, 1])

# plt.subplot(3, 1, 1)
plt.plot(x[:,0], x[:,1], linewidth=1, c='r')
plt.plot(X0[0], X0[1], 'ro', Xf[0], Xf[1], 'ro')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
ax = plt.gca()
circle1 = plt.Circle((circle[0], circle[1]), circle[2], color='r')
# ax.add_patch(circle1)
# ax.set_aspect('equal', adjustable='box')
plt.show()
mp.plt.show()