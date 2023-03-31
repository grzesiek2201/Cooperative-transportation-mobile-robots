import numpy as np


class Controller:
    def __init__(self, k1=120, k2=120):
        self.pc = 0
        self.k1 = k1
        self.k2 = k2
        self.prev_qr = [0, 0]
    
    def control(self, qr, pr):
        pass

    def scr(self, e, q, qr):
        """Sliding control rule

        Args:
            e (list[float, float, float]): pose error [x, y, theta]
            q (list[float, float]): current controls
            qr (list[float, float]): desired controls
        """
        s1 = e[0]
        s2 = e[2] + np.arctan(qr[0]*e[2])
        ar = qr[0] - self.prev_qr[0]  # vr prim
        v = e[1]*q[1] + qr[0]*np.cos(e[2]) + self.k1*sat(s1)
        w = qr[1] + (e[1] / (1+(qr[0]*e[1])**2) * ar) + (qr[0] / (1+(qr[0]*e[1])**2) * qr[0]) * np.sin(e[2]) + self.k2*sat(s2)
        return v, w


def sat(s):
    if s > 1:
        return 1
    elif s < -1:
        return -1
    else:
        return s
    

if __name__ == '__main__':
    controller = Controller()
    pass
    pass