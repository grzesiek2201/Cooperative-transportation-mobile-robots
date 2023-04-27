import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.patches import Rectangle


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

    if round(s1[1], 8) == round(s2[1], 8) or round(s1[1], 8) == round(s3[1], 8) or round(s1[1], 8) == round(s4[1], 8):  # the rectangle is horizontal or vertical
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

    else:

        lines = [Line(s1, s2), Line(s2, s3), Line(s3, s4), Line(s4, s1)]  # lines between corners (edges) in order
        p_max = max(rect, key=lambda x: x[1])
        p_min = min(rect, key=lambda x: x[1])
        pix.append([math.floor(p_max[0] / res) if p_max[0] > 0 else math.floor(p_max[0] / res), round(p_max[1] / res)])
        pix.append([math.floor(p_min[0] / res) if p_min[0] > 0 else math.floor(p_min[0] / res), round(p_min[1] / res)])

        for line in lines:
            pts = line.p1, line.p2
            pts_y = line.p1[1], line.p2[1]
            ytop_id = pts_y.index(max(pts_y))  # max y value id for the edge
            ytop = pts[ytop_id][1]  # / res  # math.ceil(pts[ytop_id][1] / res)  # max y value with respect to resolution
            ybot = pts[int(not ytop_id)][1]  # / res # math.floor(pts[int(not ytop_id)][1] / res)  # min y value for the edge

            while ytop > ybot:
                x = line.calculate_x(ytop)  # x value for this edge
                x = math.ceil(x / res) if x > 0 else math.floor(x / res)
                # y = math.ceil(ytop / res) if ytop > 0 else math.floor(ytop / res)
                pix.append([x, round(ytop / res)])  # add the max/min x for this edge to the dictionary
                ytop -= res  # go lower
        
        pix = np.array(pix)
        

        # plt.scatter(pix[:, 0], pix[:, 1])
        # plt.show()

        # fill
        ys = np.unique(pix[:, 1])
        for y in ys:
            boundaries = pix[pix[:, 1] == y]  # boundaries for the given y level
            x_min = min(boundaries[:, 0])
            x_max = max(boundaries[:, 0])
            if x_min == x_max:  # up or down corner
                pixies.append([x_min, y])
                # for x in range(int(x_min), int(x_min)):
                #     pixies.append([x, math.ceil(abs(y))*math.copysign(1, y)])
            else:
                for x in range(int(x_min), int(x_max)):
                    pixies.append([x, y])
    # print(pixies)
    return pixies

def main():
    x, y, theta = 0, 0, 0
    width, height = 5.1, 4.01  # [m]
    res = .25  # [m]

    trajectory = [[0, 0, 0], [4, 1, math.pi/16], [6, 2, math.pi/8], [10, 5, math.pi/4], [11, 8, math.pi*3/8], [12, 12, math.pi/2]]

    rect = vert_from_params((x, y, theta, width/2, height/2))  # get corners from position and width, height
    pixies = np.array([[0, 0]])
    for x, y, theta in trajectory:  
        temp_rect = translate_corners(rect, (x, y))
        temp_rect = rotate_around_point(temp_rect, theta, (x, y))
        pixies = np.vstack((pixies, pix_from_cont(temp_rect, res=res)))

    print(len(pixies))
    pixies = set(map(tuple, pixies))
    print(len(pixies))


    fig = plt.figure()
    ax = plt.gca()

    for pix in pixies:
        ax.add_patch(Rectangle((pix[0], pix[1]-.5*res), 1*.9, 1*.9, 0))
        # plt.plot(pix[0]+.5*res, pix[1]-.5*res, 'rs')

    # for corner in rect:
    #     plt.plot(corner[0]/res, corner[1]/res, 'sk')

    ax.set_aspect('equal')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()