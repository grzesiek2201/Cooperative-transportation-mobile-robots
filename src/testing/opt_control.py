import numpy as np
import control as ct
import control.optimal as opt
import matplotlib.pyplot as plt


from weightedlpsum import wlps
import scipy

# general:
# res = obc.solve_ocp(sys, timepts, X0, cost, constraints)
#
# constraints:
# LinearConstraint(A, lb, ub)
# NonlinearConstraint(f, lb, ub)
# lb <= A @ np.hstack([x, u]) <= ub
# lb <= f(x, u) <= ub


def vehicle_update(t, x, u, params):
    # Get the parameters for the model
    l = params.get('wheelbase', 3.)         # vehicle wheelbase
    phimax = params.get('maxsteer', 0.5)    # max steering angle (rad)

    # Saturate the steering input
    phi = np.clip(u[1], -phimax, phimax)

    # Return the derivative of the state
    return np.array([
        np.cos(x[2]) * u[0],            # xdot = cos(theta) v
        np.sin(x[2]) * u[0],            # ydot = sin(theta) v
        u[1]#(u[0] / l) * np.tan(phi)        # thdot = v/l tan(phi)
    ])

def vehicle_output(t, x, u, params):
    return x                            # return x, y, theta (full state)


def sqrt_cost(x, u):
    return np.sqrt(np.power(np.cos(x[2]) * u[0], 2) +            # xdot = cos(theta) v
                   np.power(np.sin(x[2]) * u[0], 2))             # ydot = sin(theta) v


# Define the vehicle steering dynamics as an input/output system
vehicle = ct.NonlinearIOSystem(
    vehicle_update, vehicle_output, states=3, name='vehicle',
    inputs=('v', 'phi'), outputs=('x', 'y', 'theta'))

x0 = np.array([-5., 0., 0.]); u0 = np.array([0., 0.])
xf = np.array([10., 2., 0.]); uf = np.array([0., 0.])
Tf = 2

Q = np.diag([0, 0, 0])          # don't turn too sharply
R = np.diag([0, 0])               # keep inputs small
P = np.diag([1000, 1000, 1000])   # get close to final point
# traj_cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
traj_cost = sqrt_cost
# traj_cost = lambda x, u: 1
term_cost = opt.quadratic_cost(vehicle, P, 0, x0=xf)

# weighted lp norm space constraint
# lp_constraint = (scipy.optimize.NonlinearConstraint((lambda x, u: wlps(x, u, (2, 1), 10)), 1, float('inf'), keep_feasible=False))
test_constraint = (scipy.optimize.NonlinearConstraint((lambda x, u: x[0]**2 + x[1]**2), 2, np.inf))
# state_constraint = opt.state_range_constraint(vehicle, [-float('inf'), 0, -float('inf')], [float('inf'), 1.5, float('inf')])
constraints = [ opt.input_range_constraint(vehicle, [0, -3], [12, 3]), test_constraint]#, test_constraint]#, lp_constraint ]

timepts = np.linspace(0, Tf, 20, endpoint=True)
result = opt.solve_ocp(
    vehicle, timepts, x0, 
    traj_cost, 
    constraints=constraints,
    terminal_cost=term_cost, 
    initial_guess=u0,
    minimize_method='SLSQP',
    minimize_options={"maxiter": 10000, "disp": True},
    )

# Simulate the system dynamics (open loop)
resp = ct.input_output_response(
    vehicle, timepts, result.inputs, x0,
    t_eval=np.linspace(0, Tf, 100))
t, y, u = resp.time, resp.outputs, resp.inputs

# Plotting the result
plt.subplot(3, 1, 1)
plt.plot(y[0], y[1])
plt.plot(x0[0], x0[1], 'ro', xf[0], xf[1], 'ro')
plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.subplot(3, 1, 2)
plt.plot(t, u[0])
# plt.axis([0, 10, 9.9, 10.1])
plt.xlabel("t [sec]")
plt.ylabel("u1 [m/s]")

plt.subplot(3, 1, 3)
plt.plot(t, u[1])
# plt.axis([0, 10, -0.015, 0.015])
plt.xlabel("t [sec]")
plt.ylabel("u2 [rad/s]")

plt.suptitle("Lane change manuever")
plt.tight_layout()
plt.show()