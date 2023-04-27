import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class RotatingRectangle(Rectangle):
    def __init__(self, xy, width, height, rel_point_of_rot, **kwargs):
        super().__init__(xy, width, height, **kwargs)
        self.rel_point_of_rot = rel_point_of_rot
        self.xy_center = self.get_xy()
        self.set_angle(self.angle)

    def _apply_rotation(self):
        angle_rad = self.angle * np.pi / 180
        m_trans = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)]])
        shift = -m_trans @ self.rel_point_of_rot
        self.set_xy(self.xy_center + shift)

    def set_angle(self, angle):
        self.angle = angle
        self._apply_rotation()

    def set_rel_point_of_rot(self, rel_point_of_rot):
        self.rel_point_of_rot = rel_point_of_rot
        self._apply_rotation()

    def set_xy_center(self, xy):
        self.xy_center = xy
        self._apply_rotation()


if __name__ == '__main__':
    height = 0.1
    width = 1
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim([-width * 1.2, width * 1.2])
    ax.set_ylim([-width * 1.2, width * 1.2])
    ax.plot(0, 0,  color='r', marker='o', markersize=10)
    point_of_rotation = np.array([width/2, height/2])          # A
    # point_of_rotation = np.array([width/2, height/2])  # B
    # point_of_rotation = np.array([width/3, height/2])  # C
    # point_of_rotation = np.array([width/3, 2*height])  # D

    # for deg in range(0, 360, 45):
    #     rec = RotatingRectangle((0, 0), width=width, height=height, 
    #                             rel_point_of_rot=point_of_rotation,
    #                             angle=deg, color=str(deg / 360), alpha=0.9)
    #     ax.add_patch(rec)
    rec = RotatingRectangle((0, 0), width=width, height=height, 
                        rel_point_of_rot=point_of_rotation,
                        angle=20, color='black', alpha=0.9,
                        fill=None)
    ax.add_patch(rec)
    plt.show()