import math
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
from rotrectangle import RotatingRectangle


def rectangle(x, y, alpha, height, width):
    pos_x = x - width / 2
    pos_y = y - height / 2
    corner_x = pos_x * math.cos(alpha) - pos_y * math.sin(alpha)
    corner_y = pos_x * math.sin(alpha) + pos_y * math.cos(alpha)
    return corner_x, corner_y


def read_data(filename):

    try:
        path = list(Path(__file__).parent.parent.glob(f"graph-search/anytime_d_star/{filename}"))[0]
        with open(path, 'r') as f:
            data = [eval(line.strip()) for line in f]
    except FileNotFoundError as e:
        logging.error(e)
    except IndexError as e:
        logging.error(e)

    return data


if __name__ == '__main__':
    postures = read_data('path.csv')

    plt.figure(figsize=(8, 8))
    plt.scatter(0, 0)
    ax = plt.gca()
    ax.set_aspect('equal')

    fronts = [[], []]
    width, height = 6, 3
    point_of_rotation = np.array([width/2, height/2])
    for i, posture in enumerate(postures):
        # if i%2:
        #     continue
        rec = RotatingRectangle((posture[0], posture[1]), width=width, height=height, 
                        rel_point_of_rot=point_of_rotation,
                        angle=posture[2]*180.0/np.pi, color='black', alpha=0.9,
                        fill=None)
        ax.add_patch(rec)
        fronts[0].append(posture[0] + math.cos(posture[2]) * width/2)
        fronts[1].append(posture[1] + math.sin(posture[2]) * width/2)

    plt.scatter(fronts[0], fronts[1], s=4)

    plt.show()