import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    plt.figure()
    points = """\
    [(SX(-3.93446), SX(4.95402)), (SX(-3.46858), SX(6.69269)), (SX(-3.75836), SX(6.77033)), (SX(-4.22424), SX(5.03167))]\
    """
    points = points.replace("[", "")
    points = points.replace("SX", "")
    points = points.replace("]", "")
    points = points.replace("[", "")
    points = np.array(eval(points))
    x = np.hstack((points[:, 0], points[0][0]))
    y = np.hstack((points[:, 1], points[0][1]))
    plt.scatter(0, 0)
    plt.plot(x, y)
    plt.xlim((-6, 20))
    plt.ylim(-8, 20)
    plt.grid()
    plt.show()