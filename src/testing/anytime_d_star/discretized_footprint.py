import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.patches import Rectangle
from bresenham import bresenham


class Line:
    def __init__(self, p1: tuple, p2: tuple):
        self.p1 = p1
        self.p2 = p2
        self.norm = np.linalg.norm(self.p2 - self.p1)
        if round(p2[0], 4) == round(p1[0], 4):
            self.dir = float("inf")
            self.x_pos = p1[0]
            self.const_term = None
            return
        else:
            self.dir = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.const_term = self.p1[1] - self.dir * self.p1[0]

    def calculate_y(self, x0: float) -> float:
        """
        Get y value from line equation and x0
        :param x0: point at which the y should be calculated
        :return: value of y
        """
        return self.dir * x0 + self.const_term  # y=ax+b -> x = (y-b)/a

    def calculate_x(self, y0: float) -> float:
        """
        Get y value from line equation and y0
        :param y0: point at which the x should be calculated
        :return: value of x
        """
        if self.dir == 0:
            return 
        return (y0 - self.const_term) / self.dir


def translate_corners(rectangle, xy):
    return np.array([(corner[0] + xy[0], corner[1] + xy[1]) for corner in rectangle])


def rotate_origin(rectangle, angle):
    return np.array([(math.cos(angle) * corner[0] - math.sin(angle) * corner[1], 
                      math.sin(angle) * corner[0] + math.cos(angle) * corner[1]) for corner in rectangle])


def rotate_around_point(rectangle, angle, point):
    return np.array([(point[0] + (corner[0] - point[0]) * math.cos(angle) - (corner[1] - point[1]) * math.sin(angle),
                      point[1] + (corner[0] - point[0]) * math.sin(angle) + (corner[1] - point[1]) * math.cos(angle),)
                      for corner in rectangle])


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


def pix_from_cont(rect, res=1):
    # filled rectangle pixels from its corner coordiantes in continuous space based on resolution of the grid
    pix = []  # list of max and min x values for given y value
    pixies = []  # list of occupied pixels
    s1, s2, s3, s4 = rect[0], rect[1], rect[2], rect[3]  # all corners in order

    # the rectangle is horizontal or vertical
    if round(s1[1], 8) == round(s2[1], 8) or round(s1[1], 8) == round(s3[1], 8) or round(s1[1], 8) == round(s4[1], 8):
        x_min = math.floor(min(s1[0], s2[0], s3[0], s4[0]) / res)
        x_max = math.ceil(max(s1[0], s2[0], s3[0], s4[0]) / res)
        y_min = math.floor(min(s1[1], s2[1], s3[1], s4[1]) / res)
        y_max = math.ceil(max(s1[1], s2[1], s3[1], s4[1]) / res)
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                pixies.append([i, j])
        pixies = np.array(pixies)
        # plt.scatter(pixies[:, 0], pixies[:, 1])
        # plt.show()

    # the rectangle is rotated
    else:
        lines = [Line(s1, s2), Line(s2, s3), Line(s3, s4), Line(s4, s1)]  # lines between corners (edges) in order
        p_max = max(rect, key=lambda x: x[1])
        p_min = min(rect, key=lambda x: x[1])

        for line in lines:
            pts = line.p1, line.p2
            pts_y = line.p1[1], line.p2[1]
            ytop_id = pts_y.index(max(pts_y))  # max y value id for the edge
            ytop = pts[ytop_id][1]  # max y value with respect to resolution
            ybot = pts[int(not ytop_id)][1]  # min y value for the edge

            # for top corner
            y = ytop
            x = line.calculate_x(y)
            pix.append([round(x/res), round(y/res)])
            # for bottom corner
            y = ybot
            x = line.calculate_x(y)
            pix.append([round(x/res), round(y/res)])

            if round(ytop/res) - 0.5 > ybot: 
                y = round(ytop/res) - 0.5
                x = line.calculate_x(y)
                pix.append([round(x), y+.5])
                while (y-1) > ybot:
                    y -= 1
                    x = line.calculate_x(y)
                    pix.append([round(x), y+.5])
        
        pix = np.array(pix)
        # plt.scatter(pix[:, 0], pix[:, 1])
        # plt.show()

        # fill
        ys = np.unique(pix[:, 1])
        for y in ys:
            boundaries = pix[pix[:, 1] == y]  # boundaries for the given y level
            x_min = min(boundaries[:, 0])
            x_max = max(boundaries[:, 0])
            for x in range(int(x_min), int(x_max)+1):
                pixies.append([x, y])
    # print(pixies)
    return pixies


def pix_from_cont_bresenham(rect, res=1):
    # filled rectangle pixels from its corner coordiantes in continuous space based on resolution of the grid
    pix = []  # list of max and min x values for given y value
    pixies = []  # list of occupied pixels
    s1, s2, s3, s4 = rect[0], rect[1], rect[2], rect[3]  # all corners in order
    lines = [bresenham(round(s1[0]/res), round(s1[1]/res), round(s2[0]/res), round(s2[1]/res)), 
             bresenham(round(s2[0]/res), round(s2[1]/res), round(s3[0]/res), round(s3[1]/res)), 
             bresenham(round(s3[0]/res), round(s3[1]/res), round(s4[0]/res), round(s4[1]/res)), 
             bresenham(round(s4[0]/res), round(s4[1]/res), round(s1[0]/res), round(s1[1]/res))]
    for line in lines:
        pix += list(line)
    pix = np.array(pix)

    ys = np.unique(pix[:, 1])
    for y in ys:
        boundaries = pix[pix[:, 1] == y]  # boundaries for the given y level
        x_min = min(boundaries[:, 0])
        x_max = max(boundaries[:, 0])
        for x in range(x_min, x_max+1):
            pixies.append([x, y])
    return pixies


def get_conf_turn(radius, n_samples, start_angle, stop_angle, direction):
    # start_angle %= (2*math.pi)
    # stop_angle %= (2*math.pi)
    if direction == "cw":

        start_angle += math.pi/2
        stop_angle += math.pi/2

        x_transf = radius * math.cos(start_angle)
        y_transf = radius * math.sin(start_angle)

        angles = np.linspace(start_angle, stop_angle, n_samples)
        conf = []

        for angle in angles:
            x = radius * math.cos(angle) - x_transf
            y = radius * math.sin(angle) - y_transf
            theta = -(math.pi/2 - angle)
            conf.append((x, y, theta))
            
    elif direction == "ccw":
        start_angle -= math.pi/2
        stop_angle -= math.pi/2

        x_transf = radius *  math.cos(start_angle)
        y_transf = radius * math.sin(start_angle)

        angles = np.linspace(start_angle, stop_angle, n_samples)
        conf = []

        for angle in angles:
            x = radius * math.cos(angle) - x_transf
            y = radius * math.sin(angle) - y_transf
            # theta = math.pi/2 + angle
            theta = -(math.pi/2 - angle)
            conf.append((x, y, theta))  
    
    return conf


def orient_from_key(key):
    if key == "0pi":
        return 0
    elif key == "1/2pi":
        return math.pi/2
    elif key == "pi":
        return math.pi
    elif key == "3/2pi":
        return math.pi*3/2
    elif key == "1/4pi":
        return math.pi/4
    elif key == "3/4pi":
        return math.pi*3/4
    elif key == "5/4pi":
        return math.pi*5/4
    elif key == "7/4pi":
        return math.pi*7/4


def get_footprints(mps, width=2.99, height=0.99, res=1):
    x, y, theta = 0, 0, 0
    width, height = width, height  # [m]
    res = res  # [m]

    rect = vert_from_params((x, y, theta, width/2, height/2))  # get corners from position and width, height

    mps = mps

    footprints = {}

    for orientation, motions in mps.items():  # for each orientation
        # print(orientation, motions)
        footprints[orientation] = []
        pixies = None
        samples = 30

        for motion in motions:  # for each motion in orientation
            orient = orient_from_key(orientation)
            pixies = np.array([[]]).reshape(-1, 2)
            if motion[2] != 0:  # if there's rotation (turning)
                dir = "cw" if math.copysign(1, motion[2]) == -1 else "ccw"  # if sign positive -> move ccw, else cw
                trajectory = get_conf_turn(radius = abs(motion[0]), n_samples=30, start_angle=orient, stop_angle = orient+motion[2], direction=dir)
            
            elif abs(motion[0]) and abs(motion[1]):  # diagonal line
                inter_xs = [x for x in range(0, int(motion[0] + 1*math.copysign(1, motion[0])), int(math.copysign(1, motion[0])))]
                inter_ys = [y for y in range(0, int(motion[1] + 1*math.copysign(1, motion[1])), int(math.copysign(1, motion[1])))]
                trajectory = [[x, y, orient] for x, y in zip(inter_xs, inter_ys)]
            
            elif abs(motion[0]):  # horizontal line
                trajectory = [[x, 0, orient] for x in range(0, int(motion[0] + 1*math.copysign(1, motion[0])), int(math.copysign(1, motion[0])))]
            
            elif abs(motion[1]):  # vertical line
                trajectory = [[0, y, orient] for y in range(0, int(motion[1] + 1*math.copysign(1, motion[1])), int(math.copysign(1, motion[1])))]
            
            for x, y, theta in trajectory:  
                temp_rect = translate_corners(rect, (x, y))
                temp_rect = rotate_around_point(temp_rect, theta, (x, y))
                pixies = np.vstack((pixies, pix_from_cont_bresenham(temp_rect, res)))
            footprints[orientation].append(set(map(tuple, pixies)))
            # footprints[orientation].append(np.array(list(set(map(tuple, pixies)))))
    
    return footprints


def main():
    x, y, theta = 0, 0, 0
    width, height = 3, 1  # [m]
    res = .25  # [m]

    # trajectory = [[0, 0, 0], [-2, 0, 0]]#[4, 0, math.pi/16], [8, 2, math.pi/8], [10, 5, math.pi/4], [12, 8, math.pi*3/8], [12, 12, math.pi/2]]
    # trajectory = [[0, 0, 0], [2, 2, math.pi/2]]   
    trajectory = get_conf_turn(radius=5, n_samples=30, start_angle=0, stop_angle=math.pi/2, direction='ccw')

    rect = vert_from_params((x, y, theta, width/2, height/2))  # get corners from position and width, height
    pixies = np.array([[]]).reshape(-1, 2)
    temp_rect = None
    for x, y, theta in trajectory:  
        temp_rect = translate_corners(rect, (x, y))
        temp_rect = rotate_around_point(temp_rect, theta, (x, y))
        # pixies = np.vstack((pixies, pix_from_cont(temp_rect, res=res)))
        pixies = np.vstack((pixies, pix_from_cont_bresenham(temp_rect, res)))

    pixies = set(map(tuple, pixies))
    print(len(pixies))
    print(pixies)

    
    ax = plt.gca()
    ax.set_aspect('equal')
    for pix in pixies:
        ax.add_patch(Rectangle((pix[0]-.5*res, pix[1]-.5*res), 1*.9, 1*.9, 0))

    for corner in temp_rect:
        plt.plot(corner[0]/res, corner[1]/res, 'sk')

    ax.set_aspect('equal')
    plt.grid()
    plt.title("Discretized robot footprint")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == '__main__':
    main()
    # from env import Env
    # env = Env(5, 5)
    # WIDTH = 2.99
    # HEIGHT = 0.99
    # res = .25
    # footprints = get_footprints(mps=env.motions_pi_backwards, width=WIDTH, height=HEIGHT, res=res)
    # for pixies in footprints.values():
    #     # pixies = footprints["0pi"]
        
    #     for pixers in pixies:
    #         plt.figure()
    #         ax = plt.gca()
    #         for pix in pixers:
    #             ax.add_patch(Rectangle((pix[0]-.5*res, pix[1]-.5*res), 1*.9, 1*.9, 0))
    #         ax.set_aspect('equal')
    #         plt.show()

