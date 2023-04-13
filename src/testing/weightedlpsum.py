import numpy as np
import matplotlib.pyplot as plt


# def wlps(x, u, s, p, n=2):
#     # XY = np.vstack(list(zip(x,y)))
#     xy = np.stack((x[0], x[1]))
#     norm = np.power(np.sum(np.power(np.abs(xy)/s, p)), 1/p)
#     # if norm < 1:
#         # print(f"{x=}")
#     return norm

def wlps(xy, u, s, p, n=2):
    # XY = np.vstack(list(zip(x,y)))
    # xy=(x[0], x[1])
    # norm = ((abs(x[0])/s[0])**p + (abs(x[1])/s[1])**p)**(1/p)
    norm = np.power(np.sum(np.power(np.abs(xy)/s, p)), 1/p)
    # if norm < 1:
        # print(f"{x=}")
    return norm

def wlpn(x, s, p, h, k, a):
    """
    Weighted Lp norm for R2 with translation and rotation

    Args:
        x (list[float]): values to compute
        s (list[float]): sigma vector (vector of weights) 
        p (float): p parameter of the Lp norm
        h (float): horizontal shift, positive value shifts to the right
        k (float): vertical shift, positive value shifts up
        a (float): rotation angle in radians, positive shifts according to coordinate system handedness

    Returns:
        float: value of the norm
    """
    return np.power(np.abs((x[0] - h)*np.cos(a)+(x[1]-k)*np.sin(a)) / s[0], p) +\
    np.power(np.abs((x[0] - h)*np.sin(a)-(x[1]-k)*np.cos(a)) / s[1], p)

def wlps_space(xy, s, p, n=2):
    # XY = np.vstack(list(zip(x,y)))
    norm = np.power(np.sum(np.power(np.abs(xy)/s, p), axis=1), 1/p)
    return norm


def circle(xy, r):
    return (xy[0]**2 + xy[1]**2 - r**2)


RESOLUTION = (200, 200)
XY_RANGE = [[-5., 5.], [-5., 5.]]
S = np.array([2, 1])
P = 20
POINT_SIZE = 2
R = .5
E = 0.01

def main():
    X = np.linspace(XY_RANGE[0][0], XY_RANGE[0][1], RESOLUTION[0])
    Y = np.linspace(XY_RANGE[1][0], XY_RANGE[1][1], RESOLUTION[1])
    s = S + R + E
    p = P
    xx, yy = np.meshgrid(X, Y)
    XY = np.stack((xx.ravel(), yy.ravel()), axis=1)
    result = np.array([xy for xy in XY if wlps(xy, 0, s, p) - 1 > 0]).reshape(-1, 2)
    # result = np.array([xy for xy in XY if circle(xy, r=1) > 0]).reshape(-1, 2)
    plt.scatter(result[:, 0], result[:, 1])
    # result = wlps_space(XY, s, p)
    # mask = result > 1
    # plt.scatter(np.vstack((XY[mask][:,0], XY[mask][:,1]))[0], np.vstack((XY[mask][:,0], XY[mask][:,1]))[1], s=POINT_SIZE)
    plt.xlim(XY_RANGE[0][0], XY_RANGE[0][1])
    plt.ylim(XY_RANGE[1][0], XY_RANGE[1][1])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(which='major', color='gray', linestyle='--', linewidth=.22)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=.22)
    plt.plot([-S[0], S[0], S[0], -S[0], -S[0]], [S[1], S[1], -S[1], -S[1], S[1]])
    plt.show()



if __name__ == '__main__':
    main()