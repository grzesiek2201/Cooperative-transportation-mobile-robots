import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rotrectangle import RotatingRectangle
from pathlib import Path
import logging
import math


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
    def __init__(self, d: float=0, alpha: float=0, x0: np.ndarray=np.array([0, 0, 0])):
        self.d = d
        self.alpha = alpha
        self.state_ref = x0 + [self.d * math.cos(x0[2]), self.d * math.sin(x0[2]), 0]
        self.state_ref_prev = x0 + [self.d * math.cos(x0[2]), self.d * math.sin(x0[2]), 0]
        self.control = None

    def update_vel(self, ts):
        # updated reference velocities in order to input them into controller
        # x_prev is previous position of follower, x is current position of follower
        vx = (self.state_ref[0] - self.state_ref_prev[0]) / ts
        vy = (self.state_ref[1] - self.state_ref_prev[1]) / ts
        v = np.sqrt(vx * vx + vy * vy)
        w =  (self.state_ref[2] - self.state_ref_prev[2]) / ts # dont know where to get it from
        if abs(w) > math.pi:
            w = (self.state_ref[2]%(2*math.pi) - self.state_ref_prev[2]%(2*math.pi)) / ts
        self.control = np.array([float(v), float(w)])

    def update_pos(self, x_vl, x_vl_prev, u_vl):
        # x_vl is the state vector of Virtual Leader at time step k+1
        x_new = x_vl[0] + self.d * np.cos(self.alpha + x_vl_prev[2])
        y_new = x_vl[1] + self.d * np.sin(self.alpha + x_vl_prev[2])
        if u_vl[1] != 0:
            theta_new = np.arctan2((y_new - self.state_ref[1]), (x_new - self.state_ref[0]))
        else:
            theta_new = x_vl_prev[2]
        self.state_ref_prev = self.state_ref
        self.state_ref = np.array([float(x_new), float(y_new), float(theta_new)])


def rectangle(x, y, alpha, height, width):
    pos_x = x - width / 2
    pos_y = y - height / 2
    corner_x = pos_x * np.cos(alpha) - pos_y * np.sin(alpha)
    corner_y = pos_x * np.sin(alpha) + pos_y * np.cos(alpha)
    return corner_x, corner_y


def read_data(filename='trajectory.csv'):
    try:
        path = list(Path(__file__).parent.parent.glob(f"op-mp/{filename}"))[0]
        data = pd.read_csv(path)
    except FileNotFoundError as e:
        logging.error(e)
    except IndexError as e:
        logging.error(e)

    # data = data[data["v"]]
    vel_dif_threshold = 0.2
    v_diff = data["v"][1:].to_numpy() - data["v"][:-1].to_numpy()
    idx = np.where(v_diff > vel_dif_threshold)

    # for id in idx:
    #     data = data.drop(id)  

    X = np.array(data["x"].to_list()).reshape(-1, 1)
    Y = np.array(data["y"].to_list()).reshape(-1, 1)
    Theta = np.array(data["0"].to_list()).reshape(-1, 1)
    U = np.hstack((np.array(data["v"].to_list()).reshape(-1, 1), np.array(data["w"].to_list()).reshape(-1, 1)))
    for id in idx:
        U[id] = (U[id-1] + U[id+1]) / 2
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
    for i in range(len(X)):
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


def save_history(filename, x, y, t, v, w):
    try:
        path = list(Path(__file__).parent.parent.glob(f"op-mp/{filename}"))[0]
    except IndexError:
        print(f"File {filename} does not exist")
        exit()
    df = pd.DataFrame(np.hstack((np.array(x).reshape(-1, 1), 
                                 np.array(y).reshape(-1, 1),
                                 np.array(t).reshape(-1, 1),
                                 np.array(v).reshape(-1, 1),
                                 np.array(w).reshape(-1, 1))))
    df.columns = ["x", "y", "t", "v", "w"]
    df.to_csv(path, index=False)


def main():
    X, Y, Theta, U, T = read_data()
    follower1 = Follower(d=0.35, alpha=math.pi/2, x0=np.array([float(X[0]), float(Y[0]), float(Theta[0])]))
    follower2 = Follower(d=-0.35, alpha=math.pi/2, x0=np.array([float(X[0]), float(Y[0]), float(Theta[0])]))
    x_prev = [X[0], Y[0], Theta[0]]
    follower1_history_x = []
    follower1_history_y = []
    follower1_history_t = []
    follower1_history_v = []
    follower1_history_w = []
    follower2_history_x = []
    follower2_history_y = []
    follower2_history_t = []
    follower2_history_v = []
    follower2_history_w = []

    robot_size = [1.99, 0.99]

    leader = VitrualLeader()
    leader.x = [float(X[0]), float(Y[0]), float(Theta[0])]
    leader_pose_x = []
    leader_pose_y = []
    leader_pose_theta = []
    leader_v = []
    leader_w = []
    t0 = T[0]

    for i in range(len(X)-1):
            # get parameters for given time point
            u = U[i]
            t = T[i+1]
            ts = t - t0
            # previous state
            # x_prev = leader.x
            x_prev = leader.x
            # leader.state_ref_prev = leader.state_ref
            # update the leader position based on control vector and time period
            leader.x = np.array([X[i+1], Y[i+1], Theta[i+1]])  # the updating didn't work and there was an error on the path, so it's now directly copied from trajectory
            # leader.update_pos(u[0], u[1], ts)
            # update followers position
            follower1.update_pos(leader.x, x_prev, u)
            follower2.update_pos(leader.x, x_prev, u)
            # update followers velocity
            follower1.update_vel(ts)
            follower2.update_vel(ts)
            # append all the data to lists in order to analyze
            leader_pose_x.append(leader.x[0])
            leader_pose_y.append(leader.x[1])
            leader_pose_theta.append(leader.x[2])
            leader_v.append(u[0])
            leader_w.append(u[1])
            follower1_history_x.append(follower1.state_ref[0])
            follower1_history_y.append(follower1.state_ref[1])
            follower1_history_t.append(follower1.state_ref[2])
            follower1_history_v.append(follower1.control[0])
            follower1_history_w.append(follower1.control[1])
            follower2_history_x.append(follower2.state_ref[0])
            follower2_history_y.append(follower2.state_ref[1])
            follower2_history_t.append(follower2.state_ref[2])
            follower2_history_v.append(follower2.control[0])
            follower2_history_w.append(follower2.control[1])

            t0 = t

    save_history("follower1.csv", follower1_history_x, follower1_history_y, follower1_history_t, follower1_history_v, follower1_history_w)
    save_history("follower2.csv", follower2_history_x, follower2_history_y, follower2_history_t, follower2_history_v, follower2_history_w)

    # PLOTTING 
    # # a bit of cheating
    # for i in range(1, len(follower1_history_v)):
    #     if abs(follower1_history_v[i] - follower1_history_v[i-1]) > 0.4:
    #         follower1_history_v[i] = follower1_history_v[i-1] 
    # for i in range(1, len(follower2_history_v)):
    #     if abs(follower2_history_v[i] - follower2_history_v[i-1]) > 0.4:
    #         follower2_history_v[i] = follower2_history_v[i-1] 

    # path plot
    pathfig = plt.figure(figsize=(7, 7))
    pathax = pathfig.gca()
    pathax.plot(follower1_history_x, follower1_history_y, color='blue')
    pathax.plot(follower2_history_x, follower2_history_y, color='green')
    pathax.plot(X, Y, color='red')
    pathax.grid()
    pathax.set_xlabel("x")
    pathax.set_ylabel("y")
    pathax.set_title("Transported object's and robots' paths")
    pathax.set_aspect('equal')
    pathax.legend(["robot1", "robot2", "object"])

    # plot robot's footprint
    n_arr = int(len(X) / 10)
    postures = [(X[i], Y[i], Theta[i]) for i in range(0, len(X), n_arr)]
    width, height = robot_size[0], robot_size[1]
    point_of_rotation = np.array([width/2, height/2])
    for posture in postures:
        rec = RotatingRectangle((posture[0][0], posture[1][0]), width=width, height=height, 
                        rel_point_of_rot=point_of_rotation,
                        angle=posture[2][0]*180.0/np.pi, color='black', alpha=0.9,
                        fill=None)
        
        pathax.add_patch(rec)
    plt.show()


    # X Y and Theta plots
    xytfig = plt.figure(figsize=(7, 7))
    xytsub = xytfig.subplots(3, 1)

    xytsub[0].plot(T[:-1], follower1_history_x, color='blue')
    xytsub[0].plot(T[:-1], follower2_history_x, color='green')
    xytsub[0].plot(T, X, color='red')
    xytsub[0].set_title("x")
    xytsub[0].grid()
    xytsub[0].legend(["robot1", "robot2", "object"])


    xytsub[1].plot(T[:-1], follower1_history_y, color='blue')
    xytsub[1].plot(T[:-1], follower2_history_y, color='green')
    xytsub[1].plot(T, Y, color='red')
    xytsub[1].set_title("y")
    xytsub[1].grid()
    xytsub[1].legend(["robot1", "robot2", "object"])

    xytsub[2].plot(T[:-1], follower1_history_t, color='blue')
    xytsub[2].plot(T[:-1], follower2_history_t, color='green')
    xytsub[2].plot(T, Theta, color='red')
    xytsub[2].set_title("$\Theta$")
    xytsub[2].grid()
    xytsub[2].legend(["robot1", "robot2", "object"])


    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.show()


    # v and w plot
    vwfig = plt.figure()
    vwsub = vwfig.subplots(2, 1)

    vwsub[0].plot(T[:-1], leader_v, color='red')
    vwsub[0].plot(T[:-1], follower1_history_v, color='blue')
    vwsub[0].plot(T[:-1], follower2_history_v, color='green')
    vwsub[0].set_title("v")
    vwsub[0].grid()
    vwsub[0].legend(["robot1", "robot2", "object"])

    vwsub[1].plot(T[:-1], leader_w, color='red')
    vwsub[1].plot(T[:-1], follower1_history_w, color='blue')
    vwsub[1].plot(T[:-1], follower2_history_w, color='green')
    vwsub[1].set_title("w")
    vwsub[1].grid()
    vwsub[1].legend(["robot1", "robot2", "object"])

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.show()


    # object U function plot

    fig = plt.figure()
    subplots = fig.subplots(2, 1)
    fig.suptitle("Control function for the object")
    subplots[0].plot(T, U[:,0], linewidth=2)
    subplots[0].set_ylabel("$v$")
    subplots[0].set_xlabel("t [s]")
    subplots[0].grid()
    subplots[1].plot(T, U[:,1], linewidth=2)
    subplots[1].set_ylabel("$\omega$")
    subplots[1].set_xlabel("t [s]")
    subplots[1].grid()

    # set the spacing between subplots
    plt.subplots_adjust(left=0.15,
                        bottom=0.1,
                        right=0.9,
                        top=.85,
                        wspace=0.4,
                        hspace=0.4)

    plt.show()


if __name__ == '__main__':
    main()
    # track_leader()