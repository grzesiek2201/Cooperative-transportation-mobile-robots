"""
Env 2D
@author: huiming zhou
"""
import math


class Env:
    def __init__(self):
        self.x_range = 51  # size of background
        self.y_range = 31
        self.ori_res = 8
        self.orientations = [2 * math.pi / self.ori_res * res for res in range(self.ori_res)]
        # self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
        #                 (1, 0), (1, -1), (0, -1), (-1, -1)]
        # current working version
        self.cost_f = 1.0
        self.cost_b = 5.0
        self.cost_diag = 2.0
        self.cost_arc = 10
        self.cost_rot = 5
        self.motions_pi_backwards = {
                                        "0pi":      [(1, 0, 0, self.cost_f),    (-1, 0, 0, self.cost_b),    (1, -1, -math.pi/2, self.cost_arc),     (1, 1, math.pi/2, self.cost_arc),       
                                                     (0, 0, math.pi/4, self.cost_rot), (0, 0, -math.pi/4, self.cost_rot),   (1, -1, -math.pi/4, self.cost_arc),     (1, 1, math.pi/4, self.cost_arc)],
                                        "1/2pi":    [(0, 1, 0, self.cost_f),    (0, -1, 0, self.cost_b),    (1, 1, -math.pi/2, self.cost_arc),      (-1, 1, math.pi/2, self.cost_arc),      
                                                     (0, 0, math.pi/4, self.cost_rot), (0, 0, -math.pi/4, self.cost_rot),   (1, 1, -math.pi/4, self.cost_arc),      (-1, 1, math.pi/4, self.cost_arc)],
                                        "pi":       [(-1, 0, 0, self.cost_f),   (1, 0, 0, self.cost_b),     (-1, 1, -math.pi/2, self.cost_arc),     (-1, -1, math.pi/2, self.cost_arc),     
                                                     (0, 0, math.pi/4, self.cost_rot), (0, 0, -math.pi/4, self.cost_rot),   (-1, 1, -math.pi/4, self.cost_arc),     (-1, -1, math.pi/4, self.cost_arc)],
                                        "3/2pi":    [(0, -1, 0, self.cost_f),   (0, 1, 0, self.cost_b),     (-1, -1, -math.pi/2, self.cost_arc),    (1, -1, math.pi/2, self.cost_arc),      
                                                     (0, 0, math.pi/4, self.cost_rot), (0, 0, -math.pi/4, self.cost_rot),   (-1, -1, -math.pi/2, self.cost_arc),    (1, -1, math.pi/2, self.cost_arc)],
                                        "1/4pi":    [(-1, -1, 0, self.cost_diag),   (1, 1, 0, self.cost_diag),     (0, 0, math.pi/4, self.cost_rot),   (0, 0, -math.pi/4, self.cost_rot)],
                                        "3/4pi":    [(1, -1, 0, self.cost_diag),    (-1, 1, 0, self.cost_diag),      (0, 0, math.pi/4, self.cost_rot),   (0, 0, -math.pi/4, self.cost_rot)],
                                        "5/4pi":    [(1, 1, 0, self.cost_diag),     (-1, -1, 0, self.cost_diag),  (0, 0, math.pi/4, self.cost_rot),   (0, 0, -math.pi/4, self.cost_rot)],
                                        "7/4pi":    [(-1, 1, 0, self.cost_diag),    (1, -1, 0, self.cost_diag),  (0, 0, math.pi/4, self.cost_rot),   (0, 0, -math.pi/4, self.cost_rot)],
                                    }
            # (0, 0, math.pi/2, self.cost_rot), (0, 0, -math.pi/2, self.cost_rot),
            # (0, 0, math.pi/2, self.cost_rot), (0, 0, -math.pi/2, self.cost_rot),
            # (0, 0, math.pi/2, self.cost_rot), (0, 0, -math.pi/2, self.cost_rot),
            # (0, 0, math.pi/2, self.cost_rot), (0, 0, -math.pi/2, self.cost_rot),

        # # these work
        # self.motions_pi_backwards = {
        #                              "0pi": [(-1, 0, 0), (1, 0, 0), (-1, -1, math.pi/2), (-1, 1, -math.pi/2), (0, 0, math.pi/2), (0, 0, -math.pi/2)],
        #                              "1/2pi": [(0, -1, 0), (0, 1, 0), (1, -1, math.pi/2), (-1, -1, -math.pi/2), (0, 0, math.pi/2), (0, 0, -math.pi/2)],
        #                              "pi": [(1, 0, 0), (-1, 0, 0), (1, 1, math.pi/2), (1, -1, -math.pi/2), (0, 0, math.pi/2), (0, 0, -math.pi/2)],
        #                              "3/2pi": [(0, 1, 0), (0, -1, 0), (-1, 1, math.pi/2), (1, 1, -math.pi/2), (0, 0, math.pi/2), (0, 0, -math.pi/2)],
        #                             }
        self.motions = [(-1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0),
                        (1, 0, 0), (1, -1, 0), (0, -1, 0), (-1, -1, 0)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        for i in range(10, 21):
            obs.add((i, 15))
        # for i in range(10, 21):
        #     obs.add((i, 16))
        for i in range(15):
            obs.add((20, i))
        # for i in range(15):
        #     obs.add((21, i))

        for i in range(15, 30):
            obs.add((30, i))
        # for i in range(15, 30):
        #     obs.add((31, i))
        for i in range(16):
            obs.add((40, i))
        # for i in range(16):
        #     obs.add((41, i))

        return obs
