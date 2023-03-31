import matplotlib.pyplot as plt
import numpy as np


def rotate_origin(rectangle, angle):
    return np.array([(np.cos(angle) * corner[0] - np.sin(angle) * corner[1], np.sin(angle) * corner[0] + np.cos(angle) * corner[1]) for corner in rectangle])


def rotate_around_point(rectangle, angle, point):
    return np.array([(point[0] + (corner[0] - point[0]) * np.cos(angle) - (corner[1] - point[1]) * np.sin(angle),
                      point[1] + (corner[0] - point[0]) * np.sin(angle) + (corner[1] - point[1]) * np.cos(angle),)
                      for corner in rectangle])


def translate_corners(rectangle, xy):
    return np.array([(corner[0] + xy[0], corner[1] + xy[1]) for corner in rectangle])


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


def coord_transf():
    robot = np.array([4, 2]) # position of the robot
    robot_dim = np.array([3, 1]) # width and height / 2
    xr, yr = 10, 4
    s1, s2 = 1, 2
    # obstacle corners
    rectangle = np.array([[s1, s2], [s1, -s2], [-s1, -s2], [-s1, s2]])
    rectangle = translate_corners(rectangle, [xr, yr])
    rectangle_xy = (rectangle[:, 0], rectangle[:, 1])
    # obstacle in robot's frame
    rectangle_r = translate_corners(rectangle, -robot)
    # rectangle_r = rotate_around_point(rectangle_r, np.pi/6, [xr, yr])
    rectangle_r_xy = (rectangle_r[:, 0], rectangle_r[:, 1])

    plt.figure(figsize=(7, 7))
    # plt.plot([robot[0] + robot_dim[0], robot[0] + robot_dim[0], robot[0] - robot_dim[0], robot[0] - robot_dim[0], robot[0] + robot_dim[0]], 
    #          [robot[1] + robot_dim[1], robot[1] - robot_dim[1], robot[1] - robot_dim[1], robot[1] + robot_dim[1], robot[1] + robot_dim[1]])  # robot
    plt.plot([robot_dim[0], robot_dim[0], -robot_dim[0], -robot_dim[0], robot_dim[0]], [robot_dim[1], -robot_dim[1], -robot_dim[1], robot_dim[1], robot_dim[1]])
    plt.scatter(rectangle_xy[0], rectangle_xy[1])
    plt.scatter(rectangle_r_xy[0], rectangle_r_xy[1])

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    # main()
    coord_transf()