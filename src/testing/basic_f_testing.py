import matplotlib.pyplot as plt
import numpy as np
import math


def wlps(xy, s, p):
    norm = np.power(np.sum(np.power(np.abs(xy)/s, p)), 1/p)
    return norm


def rotate_origin(rectangle, angle):
    return np.array([(math.cos(angle) * corner[0] - math.sin(angle) * corner[1], 
                      math.sin(angle) * corner[0] + math.cos(angle) * corner[1]) for corner in rectangle])


def rotate_around_point(rectangle, angle, point):
    return np.array([(point[0] + (corner[0] - point[0]) * math.cos(angle) - (corner[1] - point[1]) * math.sin(angle),
                      point[1] + (corner[0] - point[0]) * math.sin(angle) + (corner[1] - point[1]) * math.cos(angle),)
                      for corner in rectangle])


def translate_corners(rectangle, xy):
    return np.array([(corner[0] + xy[0], corner[1] + xy[1]) for corner in rectangle])


def to_rframe(rectangle, robot, deg=False):
    # transform rectangle from global to robot frame
    # deg: if robot's angle in degrees instead of radians
    translated = translate_corners(rectangle=rectangle, xy=-robot)
    if deg:
        angle = -robot[2] * np.pi / 180
    else:
        angle = -robot[2]
    transformed = rotate_origin(rectangle=translated, angle=angle)
    return transformed


def to_oframe(robot, rectangle, deg=False):
    # transform robot from global to obstacle frame
    translated = translate_corners(rectangle=robot, xy=-rectangle)
    if deg:
        angle = -rectangle[2] * np.pi / 180
    else:
        angle = -rectangle[2]
    transformed = rotate_origin(rectangle=translated, angle=angle)
    return transformed


def vert_from_params(params):
    # get rectangle vertices from its parameters
    center = params[0], params[1]
    angle = params[2]
    dimensions = params[3], params[4]

    # create the (normalized) perpendicular vectors
    v1 = np.array((math.cos(angle), math.sin(angle)))
    v2 = np.array((-v1[1], v1[0]))  # rotate by 90

    # scale them appropriately by the dimensions
    v1 *= dimensions[0]
    v2 *= dimensions[1]

    # return the corners by moving the center of the rectangle by the vectors
    return np.array([
        center + v1 + v2,
        center - v1 + v2,
        center - v1 - v2,
        center + v1 - v2,
    ])


def coord_transf_test():
    robot = np.array([4, 2, 1, 3, 1]) # parameters of the robot [x, y, theta, s1, s2]
    xo, yo = 7.4, 4
    s1, s2 = 1, 2
    rectangle_params = np.array([xo, yo, 0, s1, s2])
    # obstacle corners
    # rectangle = np.array([[s1, s2], [s1, -s2], [-s1, -s2], [-s1, s2]])
    # rectangle = translate_corners(rectangle, [xo, yo])
    rectangle = vert_from_params(rectangle_params)
    rectangle_xy = (rectangle[:, 0], rectangle[:, 1])
    # obstacle in robot's frame
    rectangle_r = to_rframe(rectangle=rectangle, robot=robot, deg=False)
    rectangle_r_xy = (rectangle_r[:, 0], rectangle_r[:, 1])

    plt.figure(figsize=(7, 7))
    # robot
    plt.scatter([robot[3], robot[3], -robot[3], -robot[3]], [robot[4], -robot[4], -robot[4], robot[4]], color='green')
    plt.arrow(robot[0], robot[1], math.cos(robot[2]), math.sin(robot[2]), length_includes_head=True, head_width=0.1)
    # obstacle in global frame
    plt.scatter(rectangle_xy[0], rectangle_xy[1], color='black')
    # obstacle in robot frame
    plt.scatter(rectangle_r_xy[0], rectangle_r_xy[1])
    plt.scatter(0, 0, color='green')
    plt.xlim(-6, 14)
    plt.ylim(-6, 8)
    plt.grid()

    # check for collision type A
    collisions_A = np.array([wlps(corner, robot[3:], p=20) for corner in rectangle_r])

    if any(collisions_A <= 1):
        print("Collision type A")

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()


    # check for collision type B
    # robot in obstacle frame
    robot_corners = vert_from_params(robot)
    robot_o = to_oframe(robot_corners, rectangle_params, deg=True)
    robot_o_xy = (robot_o[:, 0], robot_o[:, 1])

    # check for collision
    collisions_B = np.array([wlps(corner, [s1, s2], p=20) for corner in robot_o])
    if any(collisions_B <= 1):
        print("Collision type B")

    plt.figure(figsize=(7, 7))
    # robot in global frame
    plt.arrow(robot[0], robot[1], math.cos(robot[2]*np.pi/180), math.sin(robot[2]*np.pi/180), length_includes_head=True, head_width=0.1)
    # robot in obstacle frame
    plt.scatter(robot_o_xy[0], robot_o_xy[1])
    # obstacle at origin
    plt.scatter([s1, s1, -s1, -s1], [s2, -s2, -s2, s2])
    plt.scatter(0, 0, color='green')
    plt.xlim(-10, 14)
    plt.ylim(-10, 8)
    plt.grid()

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()


def main():
    s1 = 1
    s2 = 2
    rectangle = np.array([[s1, s2], [s1, -s2], [-s1, -s2], [-s1, s2]])
    rectangle_xy = (rectangle[:, 0], rectangle[:, 1])
    rectangle_rot = np.array(rotate_origin(rectangle, np.pi/8))
    rectangle_rot_xy = (rectangle_rot[:, 0], rectangle_rot[:, 1])
    rectangle_rot_tr = np.array(translate_corners(rectangle_rot, (5, 2)))
    rectangle_rot_tr_xy = (rectangle_rot_tr[:, 0], rectangle_rot_tr[:, 1])
    rectangle_rot_p = np.array(rotate_around_point(rectangle, np.pi/2, (1, 2)))
    rectangle_rot_p_xy = (rectangle_rot_p[:, 0], rectangle_rot_p[:, 1])
    plt.figure(figsize=(7,7))
    plt.scatter(rectangle_xy[0], rectangle_xy[1])
    plt.scatter(rectangle_rot_xy[0], rectangle_rot_xy[1], color='red')
    # plt.scatter(rectangle_rot_tr_xy[0], rectangle_rot_tr_xy[1], color='green')
    # plt.scatter(rectangle_rot_p_xy[0], rectangle_rot_p_xy[1], color='pink')
    plt.show()


if __name__ == '__main__':
    # main()
    coord_transf_test()