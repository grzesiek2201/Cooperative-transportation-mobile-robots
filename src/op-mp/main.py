import math
from Anytime_D_star import ADStar


def main():
    s_start = (5, 5, 0)
    s_goal = (42, 25, math.pi*3/4)
    # s_start = (4, 15, 0)
    # s_goal = (14, 25, 0)
    # s_start = (5, 5, 0)
    # s_goal = (25, 5, math.pi)
    # s_goal = (26, 5, math.pi*3/2)
    # s_goal = (8, 7, math.pi*3/4)

    # dstar = ADStar(s_start, s_goal, 1, "euclidean", 51, 31, robot_size=[1.99, 0.99], res=2)

    # # test 3
    # s_start = (5, 5, math.pi/2)
    # s_goal = (15, 11, 0)
    # dstar = ADStar(s_start, s_goal, 1, "euclidean", 30, 30, robot_size=[1.99, 0.99], res=1)

    # # test 4
    # res = 1
    # s_start = (5, 5, math.pi/2)
    # s_goal = (20, 20, 0)
    # dstar = ADStar(s_start, s_goal, 1, "euclidean", 30, 30, robot_size=[1.99, 0.99], res=res)

    # test 5 
    res = 1
    s_start = (5, 5, math.pi/2)
    s_goal = (5, 30, math.pi)
    dstar = ADStar(s_start, s_goal, 1, "euclidean", 30, 40, robot_size=[1.99, 0.99], res=res)

    # dstar = ADStar(s_start, s_goal, 1, "euclidean", 51, 31, robot_size=[1.99, 0.99], res=1)

    # dstar = ADStar(s_start, s_goal, 1, "euclidean")
    dstar.run()


if __name__ == '__main__':
    main()