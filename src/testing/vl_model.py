import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rotrectangle import RotatingRectangle


class VitrualLeader:
    def __init__(self):
        self._x = [0, 0, 0]  # state vector

    def update_pos(self, v, w, ts):  # x_prev is not needed, it's the leader's current state
        """Updates the position of the Vitrual Leader based on the discrete model of kinematics

        Args:
            v (float): linear velocity in previous time step (k)
            w (float): angular velocity in previous time step (k)
            ts (float): sampling time
        """
        x_prev = self._x
        if w != 0:
            new_theta = float(x_prev[2] + w * ts)
            new_x = float(x_prev[0] + v / w * ( np.sin(new_theta) - np.sin(x_prev[2]) ))
            new_y = float(x_prev[1] - v / w * ( np.cos(new_theta) - np.cos(x_prev[2]) ))
        if w == 0:  # basically never happens, and rounding has a negative effect on the path estimation
            new_theta = float(x_prev[2])
            new_x = float(x_prev[0] + v * ts * np.cos(x_prev[2]))
            new_y = float(x_prev[1] + v * ts * np.sin(x_prev[2]))

        self.x = [new_x, new_y, new_theta]
        return new_x, new_y, new_theta
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x):
        self._x = x


class Follower:
    def __init__(self, d=0, alpha=0):
        self.x = [0, 0, 0]
        self.x_prev = [0, 0, 0]
        self.d = d
        self.alpha = alpha
        self.u = []

    def update_vel(self, ts):
        # x_prev is previous position of follower, x is current position of follower
        vx = (self.x[0] - self.x_prev[0]) / ts
        vy = (self.x[1] - self.x_prev[1]) / ts
        v = np.sqrt(vx * vx + vy * vy)
        w = 0 # dont know where to get it from
        self.u = [v, w]

    def update_pos(self, x_vl, x_vl_prev, u_vl):
        # x_vl is the state vector of Virtual Leader at time step k+1
        x_new = x_vl[0] + self.d * np.cos(self.alpha + x_vl_prev[2])
        y_new = x_vl[1] + self.d * np.sin(self.alpha + x_vl_prev[2])
        if round(u_vl[1], 16) != 0:  # ***VERIFY THAT IT'S A SATISFYING (WORKING) SOLUTION TO DIVIDING BY ZERO IN THE ACRTAN***
            theta_new = np.arctan((y_new - self.x[1]) / (x_new - self.x[0]))
        else:
            theta_new = x_vl_prev[2]
        self.x_prev = self.x
        self.x = [x_new, y_new, theta_new]


def rectangle(x, y, alpha, height, width):
    pos_x = x - width / 2
    pos_y = y - height / 2
    corner_x = pos_x * np.cos(alpha) - pos_y * np.sin(alpha)
    corner_y = pos_x * np.sin(alpha) + pos_y * np.cos(alpha)
    return corner_x, corner_y


def read_data():
    data = pd.read_csv('D:\\Projects\\cooperative_transportation\\src\\testing\\trajectory.csv')
    X = np.array(data["x"].to_list()).reshape(-1, 1)
    Y = np.array(data["y"].to_list()).reshape(-1, 1)
    Theta = np.array(data["0"].to_list()).reshape(-1, 1)
    U = np.hstack((np.array(data["v"].to_list()).reshape(-1, 1), np.array(data["w"].to_list()).reshape(-1, 1)))
    T = np.array(data["t"].to_list()).reshape(-1, 1)
    return X, Y, Theta, U, T


def track_leader():
    X, Y, Theta, U, T = read_data()
    leader = VitrualLeader()
    leader.x = [float(X[0]), float(Y[0]), float(Theta[0])]
    leader_pose_x = []
    leader_pose_y = []
    leader_pose_theta = []
    t0 = T[0]
    for i in range(len(X)-1):
        u = U[i]
        t = T[i+1]
        ts = t - t0
        leader.update_pos(u[0], u[1], ts)
        leader_pose_x.append(leader.x[0])
        leader_pose_y.append(leader.x[1])
        leader_pose_theta.append(leader.x[2])
        t0 = t

    fig = plt.figure(figsize=(7, 7))
    subplots = fig.subplots(4, 1)
    subplots[0].plot(leader_pose_x, leader_pose_y, color='blue')
    subplots[0].plot(X[:-1], Y[:-1], color='red')
    subplots[0].legend(["estimated", "actual"])
    subplots[0].set_title("x(y)")
    subplots[1].plot(leader_pose_x, color='blue')
    subplots[1].plot(X[:-1], color="red")
    subplots[1].set_title("x")
    subplots[2].plot(leader_pose_y, color='blue')
    subplots[2].plot(Y[:-1], color="red")
    subplots[2].set_title("y")
    subplots[3].plot(leader_pose_theta, color='blue')
    subplots[3].plot(Theta[:-1], color="red")
    subplots[3].set_title("$/theta$")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def main():
    X, Y, Theta, U, T = read_data()
    follower1 = Follower(d=1, alpha=0)
    follower2 = Follower(d=-1, alpha=0)
    x_prev = [0, 0, 0]
    follower1_history_x = []
    follower1_history_y = []
    follower1_history_v = []
    follower2_history_x = []
    follower2_history_y = []
    follower2_history_v = []

    leader = VitrualLeader()
    leader.x = [float(X[0]), float(Y[0]), float(Theta[0])]
    leader_pose_x = []
    leader_pose_y = []
    leader_pose_theta = []
    t0 = T[0]

    for i in range(len(X)-1):
            u = U[i]
            t = T[i+1]
            ts = t - t0
            x_prev = leader.x
            leader.update_pos(u[0], u[1], ts)
            follower1.update_pos(leader.x, x_prev, u)
            follower2.update_pos(leader.x, x_prev, u)
            follower1.update_vel(ts)
            follower2.update_vel(ts)
            leader_pose_x.append(leader.x[0])
            leader_pose_y.append(leader.x[1])
            leader_pose_theta.append(leader.x[2])
            follower1_history_x.append(follower1.x[0])
            follower1_history_y.append(follower1.x[1])
            follower1_history_v.append(follower1.u[0])
            follower2_history_x.append(follower2.x[0])
            follower2_history_y.append(follower2.x[1])
            follower2_history_v.append(follower2.u[0])
            t0 = t

    fig = plt.figure(figsize=(7, 7))
    subplots = fig.subplots(3, 1)
    subplots[0].plot(follower1_history_x, follower1_history_y, color='blue')
    subplots[0].plot(follower2_history_x, follower2_history_y, color='green')
    subplots[0].plot(X, Y, color='red')
    subplots[0].grid()
    ax = subplots[0]
    ax.set_aspect('equal')

    # plot robot's footprint
    n_arr = int(len(X) / 20)
    postures = [(X[i], Y[i], Theta[i]) for i in range(0, len(X), n_arr)]
    width, height = 2, 1
    point_of_rotation = np.array([width/2, height/2])
    for posture in postures:
        rec = RotatingRectangle((posture[0][0], posture[1][0]), width=width, height=height, 
                        rel_point_of_rot=point_of_rotation,
                        angle=posture[2][0]*180.0/np.pi, color='black', alpha=0.9,
                        fill=None)
        ax.add_patch(rec)

    subplots[1].plot(follower1_history_x, color='blue')
    subplots[1].plot(follower2_history_x, color='green')
    subplots[1].plot(X, color='red')
    subplots[1].set_title("x")
    subplots[1].grid()

    subplots[2].plot(follower1_history_y, color='blue')
    subplots[2].plot(follower2_history_y, color='green')
    subplots[2].plot(Y, color='red')
    subplots[2].set_title("y")
    subplots[2].grid()

    plt.show()


if __name__ == '__main__':
    main()
    # track_leader()