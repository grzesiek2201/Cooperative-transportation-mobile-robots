"""
Anytime_D_star 2D
@author: huiming zhou
"""

import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

import plotting
import env

import math


from functools import wraps
from time import time

from traj_from_path import traj_from_path


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        tf = time()
        print(f"func: {f.__name__} took: {round(tf - ts, 4)} sec")
        return result
    return wrap


class ADStar:
    def __init__(self, s_start, s_goal, eps, heuristic_type, x=51, y=35, robot_size=[1, 1], res=1):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env(x, y, robot_size=robot_size, res=res)  # class Env
        self.Plot = plotting.Plotting(s_start, s_goal, x, y)

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.x = self.Env.x_range
        self.y = self.Env.y_range
        self.eg_obs = self.get_obs()

        self.g, self.rhs, self.OPEN = {}, {}, {}

        for i in range(1, self.Env.x_range - 1):
            for j in range(1, self.Env.y_range - 1):
                # self.rhs[(i, j)] = float("inf")
                # self.g[(i, j)] = float("inf")
                for k in self.Env.orientations:
                    self.rhs[(i, j, k)] = float("inf")
                    self.g[(i, j, k)] = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.eps = eps
        self.OPEN[self.s_goal] = self.Key(self.s_goal)
        self.CLOSED, self.INCONS = set(), dict()

        self.visited = set()
        self.count = 0
        self.count_env_change = 0
        self.obs_add = set()
        self.obs_remove = set()
        self.title = "Anytime D* Motion Primitives"  # Significant changes
        self.fig = plt.figure(figsize=(8, 8))

    def run(self):
        self.Plot.plot_grid(self.title)
        self.ComputeOrImprovePath()
        self.plot_visited()
        path = self.extract_path()
        self.plot_path(path)
        mps_list = self.mps_from_path(path)
        traj = traj_from_path(x0=self.s_start, path=path, path_mps=mps_list, mps=self.Env.motions_pi_backwards, vmax=1, wmax=1, a=1, e=.5, res=1)
        self.visited = set()

        while True:
            if self.eps <= 1.0:
                break
            self.eps -= 0.5
            self.OPEN.update(self.INCONS)
            for s in self.OPEN:
                self.OPEN[s] = self.Key(s)
            self.CLOSED = set()
            self.ComputeOrImprovePath()
            self.plot_visited()
            self.plot_path(self.extract_path())
            self.visited = set()
            plt.pause(0.5)

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        plt.show()

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
            print("Please choose right area!")
        else:
            self.count_env_change += 1
            x, y = int(x), int(y)
            print("Change position: s =", x, ",", "y =", y)

            # for small changes
            if self.title == "Anytime D* Motion Primitives":
                if (x, y) not in self.obs:
                    self.obs.add((x, y))
                    self.g[(x, y)] = float("inf")
                    self.rhs[(x, y)] = float("inf")
                else:
                    self.obs.remove((x, y))
                    self.UpdateState((x, y, 0))

                self.Plot.update_obs(self.obs)

                for sn, u, u_key in self.get_neighbor((x, y, 0)):
                    self.UpdateState(sn)

                plt.cla()
                self.Plot.plot_grid(self.title)

                while True:
                    if len(self.INCONS) == 0:
                        break
                    self.OPEN.update(self.INCONS)
                    for s in self.OPEN:
                        self.OPEN[s] = self.Key(s)
                    self.CLOSED = set()
                    self.ComputeOrImprovePath()
                    self.plot_visited()
                    self.plot_path(self.extract_path())
                    # plt.plot(self.title)
                    self.visited = set()

                    if self.eps <= 1.0:
                        break

            else:
                if (x, y) not in self.obs:
                    self.obs.add((x, y))
                    self.obs_add.add((x, y))
                    plt.plot(x, y, 'sk')
                    if (x, y) in self.obs_remove:
                        self.obs_remove.remove((x, y))
                else:
                    self.obs.remove((x, y))
                    self.obs_remove.add((x, y))
                    plt.plot(x, y, marker='s', color='white')
                    if (x, y) in self.obs_add:
                        self.obs_add.remove((x, y))

                self.Plot.update_obs(self.obs)

                if self.count_env_change >= 15:
                    self.count_env_change = 0
                    self.eps += 2.0
                    for s in self.obs_add:
                        self.g[(x, y)] = float("inf")
                        self.rhs[(x, y)] = float("inf")

                        for sn, u, u_key in self.get_neighbor(s):
                            self.UpdateState(sn)

                    for s in self.obs_remove:
                        for sn, u, u_key in self.get_neighbor(s):
                            self.UpdateState(sn)
                        self.UpdateState(s)

                    plt.cla()
                    self.Plot.plot_grid(self.title)

                    while True:
                        if self.eps <= 1.0:
                            break
                        self.eps -= 0.5
                        self.OPEN.update(self.INCONS)
                        for s in self.OPEN:
                            self.OPEN[s] = self.Key(s)
                        self.CLOSED = set()
                        self.ComputeOrImprovePath()
                        self.plot_visited()
                        self.plot_path(self.extract_path())
                        plt.title(self.title)
                        self.visited = set()
                        plt.pause(0.5)

            self.fig.canvas.draw_idle()

    @timing
    def ComputeOrImprovePath(self):
        while True:
            s, v = self.TopKey()
            if v >= self.Key(self.s_start) and \
                    self.rhs[self.s_start] == self.g[self.s_start]:
                break

            self.OPEN.pop(s)
            self.visited.add(s)

            if self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                self.CLOSED.add(s)
                for sn, u, u_key in self.get_neighbor(s):
                    self.UpdateState(sn)
            else:
                self.g[s] = float("inf")
                for sn, u, u_key in self.get_neighbor(s):
                    self.UpdateState(sn)
                self.UpdateState(s)

    def UpdateState(self, s):
        if s != self.s_goal:
            self.rhs[s] = float("inf")
            for x, u, u_key in self.get_neighbor(s):
                self.rhs[s] = min(self.rhs[s], self.g[x] + self.cost(s, x, u, u_key))
        if s in self.OPEN:
            self.OPEN.pop(s)

        if self.g[s] != self.rhs[s]:
            if s not in self.CLOSED:
                self.OPEN[s] = self.Key(s)
            else:
                self.INCONS[s] = 0

    def Key(self, s):
        if self.g[s] > self.rhs[s]:
            return [self.rhs[s] + self.eps * self.h(self.s_start, s), self.rhs[s]]
        else:
            return [self.g[s] + self.h(self.s_start, s), self.g[s]]

    def TopKey(self):
        """
        :return: return the min key and its value.
        """

        s = min(self.OPEN, key=self.OPEN.get)
        return s, self.OPEN[s]

    def GetKey(self):
        s = next(iter(self.OPEN))
        return s, self.OPEN[s]

    def h(self, s_start, s_goal):
        heuristic_type = self.heuristic_type  # heuristic type
        w = 0.0051

        if heuristic_type == "manhattan":
            return abs(s_goal[0] - s_start[0]) + abs(s_goal[1] - s_start[1])
        else:
            # return 1
            return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1]) + w * (s_goal[2] - s_start[2])

    def cost(self, s_start, s_goal, u, u_key):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :param u: control command
        :param u_key: control command key
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal, u, u_key):
            return float("inf")

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1]) + u[3] * 1 # u[3] is the weight of a given motion primitive

    def is_collision(self, s_start, s_end, u, u_key):
        mps = self.Env.motions_pi_backwards[u_key]  # all available motion primitives for given key
        pixies = self.Env.footprints[u_key][mps.index(u)]  # footprint of the robot for given motion primitive
        # will it work more efficient with numpy search? from what i tested so far - no
        # pixies = pixies + s_start[:2]
        # for pix in pixies:
        #     if (pix[0], pix[1]) in self.obs:
        #         return True
        for pix in pixies:
            pixt = pix[0] + s_start[0], pix[1] + s_start[1]
            if pixt in self.obs:
                return True
        return False

    def get_neighbor(self, s):
        nei_list = set()
        u_set, u_key = self.get_motion_primitives(s)
        for u in u_set:
            s_new = (s[0] + u[0], s[1] + u[1], (s[2] + u[2]) % (2*math.pi))
            if 0 < s_new[0] < self.Env.x_range - 1 and 0 < s_new[1] < self.Env.y_range - 1:
                s_next = s_new
            else:
                s_next = self.eg_obs

            if s_next not in self.obs:
                nei_list.add((s_next, u, u_key))

        return nei_list

    def get_motion_primitives(self, s):
        # motion primitives available based on current orientation
        if s[2] == 0:
            u = self.Env.motions_pi_backwards["0pi"], "0pi"
        elif s[2] == math.pi/2:
            u = self.Env.motions_pi_backwards["1/2pi"], "1/2pi"
        elif s[2] == math.pi:
            u = self.Env.motions_pi_backwards["pi"], "pi"
        elif s[2] == math.pi*3/2:
            u = self.Env.motions_pi_backwards["3/2pi"], "3/2pi"
        elif s[2] == math.pi*1/4:
            u = self.Env.motions_pi_backwards["1/4pi"], "1/4pi"
        elif s[2] == math.pi*3/4:
            u = self.Env.motions_pi_backwards["3/4pi"], "3/4pi"
        elif s[2] == math.pi*5/4:
            u = self.Env.motions_pi_backwards["5/4pi"], "5/4pi"
        elif s[2] == math.pi*7/4:
            u = self.Env.motions_pi_backwards["7/4pi"], "7/4pi"
            
        return u

    def get_obs(self):
        # get one element from obstacle set
        for e in self.obs:
            break
        return e

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_start]
        s = self.s_start

        for i in range(1000):
            g_list = {}
            for x, u, u_key in self.get_neighbor(s):
                if not self.is_collision(s, x, u, u_key):
                    g_list[x] = self.g[x]
            s = min(g_list, key=g_list.get)
            path.append(s)
            if s == self.s_goal:
                break

        # save path to file
        with open("D:\\Projects\\cooperative_transportation\\src\\testing\\anytime_d_star\\path.csv", 'w') as f:
            for state in list(path):
                f.write(str(state[0]) + ', ' + str(state[1]) + ', ' + str(state[2]) + '\n')

        return list(path)

    def plot_path(self, path):
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        pt = [x[2] for x in path]
        plt.plot(px, py, linewidth=2)
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")
        
        xy = [(x, y, t) for x, y, t in zip(px, py, pt)]
        print(xy)

    def mps_from_path(self, path):
        # get motion primitives according to points in path
        ps = [(x[0], x[1], x[2]) for x in path]
        mps_all = self.Env.motions_pi_backwards
        path_mps = []
        for i in range(len(ps)-1):
            # maybe only using the move and not key and index of motion primitive would be sufficient, might be worth to try it
            move = tuple(ps[i+1][j] - ps[i][j] for j in range(3))  # difference in configuration between two neighboring points
            u, u_key = self.get_motion_primitives(ps[i])
            mps = [mps_all[u_key][k][:3] for k in range(len(mps_all[u_key]))]  # get rid of the cost for motion primitives to allow for search in the next line
            # wrap the change in angle to [-pi, pi]
            if move[2] >= math.pi:
                move = (move[0], move[1], -(-move[2] % math.pi))
                # move[2] % math.pi * math.copysign(move[2])
            elif move[2] <= -math.pi:
                move = (move[0], move[1], move[2] % math.pi)
            mp = mps.index(move)
            path_mps.append((u_key, mp))

        # # DEBUGGING
        # # check if path_mps is correct
        # conf = self.s_start
        # pathv2 = [self.s_start]
        # for mp in path_mps:
        #     mp = self.Env.motions_pi_backwards[mp[0]][mp[1]]
        #     pathv2.append((conf[0] + mp[0], conf[1] + mp[1], (conf[2] + mp[2]) % (2*math.pi)))
        #     conf = pathv2[-1]
        # for i in range(len(pathv2)):
        #     if path[i] != pathv2[i]:
        #         print(i)
        return path_mps

    def plot_visited(self):
        self.count += 1
        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        if self.count >= len(color) - 1:
            self.count = 0

        for x in self.visited:
            plt.plot(x[0], x[1], marker='s', color=color[self.count])


def main():
    s_start = (5, 5, 0)
    s_goal = (42, 25, math.pi*3/4)
    # s_start = (4, 15, 0)
    # s_goal = (14, 25, 0)
    # s_start = (5, 5, 0)
    # s_goal = (25, 5, math.pi)

    dstar = ADStar(s_start, s_goal, 2.5, "euclidean", 51, 31, robot_size=[5.99, 2.99], res=1)
    # dstar = ADStar(s_start, s_goal, 1, "euclidean")
    dstar.run()


if __name__ == '__main__':
    main()
