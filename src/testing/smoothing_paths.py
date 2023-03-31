import matplotlib.pyplot as plt
import numpy as np


class Line:
    def __init__(self, x=[0, 1], y=[0, 1]):
        self.a = (y[1] - y[0]) / (x[1] - x[0])
        self.b = y[1] - self.a * x[1]
    
    def y_eq(self):
        # y=ax+b
        return lambda x: self.a * x + self.b

def intersection(line1, line2):
    a1, b1 = line1.a, line1.b
    a2, b2 = line2.a, line2.b
    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return x, y

def dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm((p1 - p2))

def calc_icc(x, y, radius, L2, thetas):
    # coordinates of the center of the circle
    # x = xj, y = yj
    m = 0  # direction of turn: 0 = CW, 1 = CCW
    iccx = x + (-1)**(m+1) * radius * np.sin(thetas) - L2 * np.cos(thetas)
    iccy = y + (-1)**(m) * radius * np.cos(thetas) - L2 * np.sin(thetas)
    return iccx, iccy


def main():
    x = np.array([-2, 3, 5])
    y = np.array([0, 5, 1])
    line1 = Line(x=[x[1], x[0]], y=[y[1], y[0]])
    line2 = Line(x=[x[2], x[1]], y=[y[2], y[1]])
    print(intersection(line1, line2))
    print(dist([x[0], y[0]], [x[1], y[1]]))
    theta1 = np.arctan(line1.a)  # angle between p0 and p1
    theta2 = np.arctan(line2.a)  # angle between p1 and p2
    dtheta = theta2 - theta1  # theta difference
    # R has to be taken from velocities? R = r (v / w)
    R = 1  # arbitrary R just for testing
    # thetas -> Xs, Ys -> L -> 
    thetas = theta1
    # thetas = np.arctan((y[1] - y[0]) / (x[1] - x[0]))
    L2 = R * np.tan(dtheta / 2)
    L3 = R * ( 1 / ((np.cos(dtheta / 2))) - 1)
    L1 = L3 * np.cos(dtheta / 2)
    L4 = 0
    L = L2 + L4
    iccx, iccy = calc_icc(x[1], y[1], R, L2, thetas)
    Xs = x[1] - L * np.cos(thetas)
    Ys = y[1] - L * np.sin(thetas)
    Xrs = x[1] - L2 * np.cos(thetas)
    Yrs = y[1] - L2 * np.sin(thetas)
    thetaf = np.arctan((y[1] - y[2]) / (x[1] - x[2]))
    Xrf = x[1] + L2 * np.cos(thetaf)
    Yrf = y[1] + L2 * np.sin(thetaf)
    Xf = x[1] + L * np.cos(thetaf)
    Yf = y[1] + L * np.sin(thetaf)
    # plot
    circle1 = plt.Circle((iccx, iccy), R, color='r', fill=False)
    ax = plt.gca()
    ax.add_patch(circle1)
    print(theta2 * 180.0 / np.pi)
    plt.scatter(x, y)
    plt.scatter(Xs, Ys, color='green')
    plt.scatter(Xf, Yf, color='red')
    plt.plot([x[0], Xs], [y[0], Ys], color='black')
    plt.plot([Xf, x[2]], [Yf, y[2]], color='black')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()