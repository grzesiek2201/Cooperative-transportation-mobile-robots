"""
Env 2D
@author: huiming zhou
"""
import math
from discretized_footprint import get_footprints


class Env:
    def __init__(self, x, y, robot_size=[1, 1], res=1):
        self.x_range = x  # size of background (51, 31)
        self.y_range = y
        self.ori_res = 8
        self.orientations = [2 * math.pi / self.ori_res * res for res in range(self.ori_res)]
        self.robot_size = robot_size
        self.res = res
        self.turn_r = 1*res  # turn radius
        # costs
        self.cost_f = 1.0*res
        self.cost_b = 5.0*res
        self.cost_diag = 2*res  # 1
        self.cost_arc = 5*res  # 3
        self.cost_rot = 5*res  # 3
        # *********** 45 DEGREE TURN HAS WRONG FOOTPRING ***************
        # *********** IT IS BETTER APPROXIMATED WITH DIAGONAL THAN WITH ARC ************
        self.motions_pi_backwards = {
            "0pi":      [(1.0*res, .0, .0, self.cost_f),    (-1.0*res, .0, .0, self.cost_b),    (1.0*res, -1.0*res, -math.pi/2, self.cost_arc, self.turn_r),     (1.0*res, 1.0*res, math.pi/2, self.cost_arc, self.turn_r),       
                            (.0, .0, math.pi/4, self.cost_rot), (.0, .0, -math.pi/4, self.cost_rot),
                            (2.0*res, .0, .0, 2*self.cost_f),       (-2.0*res, .0, .0, 2*self.cost_b),
                            (4.0*res, .0, .0, 4*self.cost_f),       (-4.0*res, .0, .0, 4*self.cost_b),
                            (2.0*res, -1.0*res, -math.pi/4, self.cost_arc/2),   (2.0*res, 1.0*res, math.pi/4, self.cost_arc/2)],
            
            "1/2pi":    [(.0, 1.0*res, .0, self.cost_f),    (.0, -1.0*res, .0, self.cost_b),    (1.0*res, 1.0*res, -math.pi/2, self.cost_arc, self.turn_r),      (-1.0*res, 1.0*res, math.pi/2, self.cost_arc, self.turn_r),      
                            (.0, .0, math.pi/4, self.cost_rot), (.0, .0, -math.pi/4, self.cost_rot),
                            (.0, 2.0*res, .0, 2*self.cost_f),    (.0, -2.0*res, .0, 2*self.cost_b),
                            (.0, 4.0*res, .0, 4*self.cost_f),    (.0, -4.0*res, .0, 4*self.cost_b),
                            (1.0*res, 2.0*res, -math.pi/4, self.cost_arc),      (-1.0*res, 2.0*res, math.pi/4, self.cost_arc),
                            (1, 5, 0, self.cost_f), (-1, 5, 0, self.cost_f)],
            
            "pi":       [(-1.0*res, .0, .0, self.cost_f),   (1.0*res, .0, .0, self.cost_b),     (-1.0*res, 1.0*res, -math.pi/2, self.cost_arc, self.turn_r),     (-1.0*res, -1.0*res, math.pi/2, self.cost_arc, self.turn_r),     
                            (.0, .0, math.pi/4, self.cost_rot), (.0, .0, -math.pi/4, self.cost_rot),
                            (-2.0*res, .0, .0, 2*self.cost_f),   (2.0*res, .0, .0, 2*self.cost_b),
                            (-4.0*res, .0, .0, 4*self.cost_f),   (4.0*res, .0, .0, 4*self.cost_b),
                            (-2.0*res, 1.0*res, -math.pi/4, self.cost_arc),     (-2.0*res, -1.0*res, math.pi/4, self.cost_arc)],
            
            "3/2pi":    [(.0, -1.0*res, .0, self.cost_f),   (.0, 1.0*res, .0, self.cost_b),     (-1.0*res, -1.0*res, -math.pi/2, self.cost_arc, self.turn_r),    (1.0*res, -1.0*res, math.pi/2, self.cost_arc, self.turn_r),      
                            (.0, .0, math.pi/4, self.cost_rot), (.0, .0, -math.pi/4, self.cost_rot),
                            (.0, -2.0*res, .0, 2*self.cost_f),   (.0, 2.0*res, .0, 2*self.cost_b),
                            (.0, -4.0*res, .0, 4*self.cost_f),   (.0, 4.0*res, .0, 4*self.cost_b),
                            (-1.0*res, -2.0*res, -math.pi/2, self.cost_arc),    (1.0*res, -2.0*res, math.pi/2, self.cost_arc)],
            
            "1/4pi":    [(1.0*res, 1.0*res, .0, self.cost_diag),    (-1.0*res, -1.0*res, .0, self.cost_diag),
                         (.0, .0, math.pi/4, self.cost_rot),   (.0, .0, -math.pi/4, self.cost_rot),
                         (2.0*res, 1.0*res, -math.pi/4, self.cost_arc), (1.0*res, 2.0*res, math.pi/4, self.cost_arc)],
            
            "3/4pi":    [(-1.0*res, 1.0*res, .0, self.cost_diag),   (1.0*res, -1.0*res, .0, self.cost_diag),
                         (.0, .0, math.pi/4, self.cost_rot),   (.0, .0, -math.pi/4, self.cost_rot),
                         (-1.0*res, 2.0*res, -math.pi/4, self.cost_arc), (-2.0*res, 1.0*res, math.pi/4, self.cost_arc)],
            
            "5/4pi":    [(-1.0*res, -1.0*res, .0, self.cost_diag),  (1.0*res, 1.0*res, .0, self.cost_diag),
                         (.0, .0, math.pi/4, self.cost_rot),   (.0, .0, -math.pi/4, self.cost_rot),
                         (-2.0*res, -1.0*res, -math.pi/4, self.cost_arc), (-1.0*res, -2.0*res, math.pi/4, self.cost_arc)],
            
            "7/4pi":    [(1.0*res, -1.0*res, .0, self.cost_diag),   (-1.0*res, 1.0*res, .0, self.cost_diag),
                         (.0, .0, math.pi/4, self.cost_rot),   (.0, .0, -math.pi/4, self.cost_rot),
                         (1.0*res, -2.0*res, -math.pi/4, self.cost_arc), (2.0*res, -1.0*res, math.pi/4, self.cost_arc)],
        }
        self.footprints = get_footprints(mps=self.motions_pi_backwards, width=self.robot_size[0], height=self.robot_size[1], res=self.res)

            # (.0, .0, math.pi/2, self.cost_rot), (.0, .0, -math.pi/2, self.cost_rot),
            # (.0, .0, math.pi/2, self.cost_rot), (.0, .0, -math.pi/2, self.cost_rot),
            # (.0, .0, math.pi/2, self.cost_rot), (.0, .0, -math.pi/2, self.cost_rot),
            # (.0, .0, math.pi/2, self.cost_rot), (.0, .0, -math.pi/2, self.cost_rot),

        self.motions = [(-1.0, .0, .0), (-1.0, 1.0, .0), (.0, 1.0, .0), (1.0, 1.0, .0),
                        (1.0, .0, .0), (1.0, -1.0, .0), (.0, -1.0, .0), (-1.0, -1.0, .0)]
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

        # for i in range(x):
        #     obs.add((i, 0))
        # for i in range(x):
        #     obs.add((i, y - 1))

        # for i in range(y):
        #     obs.add((0, i))
        # for i in range(y):
        #     obs.add((x - 1, i))

        # for i in range(10*self.res, 21*self.res):
        #     obs.add((i, 15*self.res))

        # for i in range(15*self.res):
        #     obs.add((20*self.res, i))

        # for i in range(15*self.res, 30*self.res):
        #     obs.add((30*self.res, i))

        # for i in range(16*self.res):
        #     obs.add((40*self.res, i))

        # for i in range(17*self.res, 22*self.res):
        #     obs.add((i, 19*self.res))

        # # test 2
        # for i in range(25, 35):
        #     for j in range(3, 10):
        #         obs.add((i, j))

        # test 3
        for i in range(0, 11*self.res): # down left hallway wall
            obs.add((13*self.res, i))

        for i in range(12*self.res+1, 30*self.res): # up left hallway wall
            obs.add((13*self.res, i))
        
        for i in range(0*self.res, 14*self.res-1):
            obs.add((i, 3*self.res))

        for i in range(4*self.res-1, 13*self.res):
            obs.add((0, i))

        for i in range(0*self.res, 13*self.res):
            obs.add((i, 13*self.res))

        # test 4 (along with test 3 obstacles)
        for i in range(0*self.res, 20*self.res):
            obs.add((17*self.res, i))
        for i in range(21*self.res+1, 30*self.res):
            obs.add((17*self.res, i))

        # test 5 (along with test 4)
        for i in range(0, 13):
            obs.add((i, 29*self.res))
        for i in range(0, 30):
            obs.add((i, 33*self.res))

        # test 6
        for i in range(18, 30):
            obs.add((i, 29))
        # obs.add((14, 28))
        # obs.add((14, 29))

        # test 7
        obs.add((7, 7))
        obs.add((8, 7))

        obs.add((15, 18))

        # test 8
        for i in range(17, 26):
            obs.add((i, 17))
        for i in range(17, 30):
            obs.add((26, i))
        

        return obs


    def export_mp(self):
        return self.motions_pi_backwards