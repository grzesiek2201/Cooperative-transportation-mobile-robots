import math
import numpy as np


class Trajectory:
    def __init__(self) -> None:
        t = []  # time values for next points
        v = []  # linear velocity at given time step
        w = []  # angular velocity at given time step
        x = []  # x position at given time step
        y = []  # y position at given time step
        theta = []  # angular position at given time step

def traj_from_path(x0, path, mps, t, vmax, wmax, a, e, res):
    # x0 - starting configuraton
    # path - list of motion primitives identifiers
    # mps - list of motion primitives
    # t - trajectory time
    # a - linear acceleration
    # e - angular acceleration
    # res - resolution
    traj = Trajectory()
    # get segments divided by motion primitives with no linear velocity
    segments = segments_from_path(path=path, mps=mps)
    # the goal is to calculate velocities for given time steps and configurations
    for type, segment in segments.values():
        if type == 'angular':
            dist = [move[2] for move in segment]
            dist = sum(dist)
        else:
            dist_lin = [math.sqrt(move[0]**2 + move[1]**2) for move in segment if move[2] == .0]
            dist_arc = [2*math.pi*move[0] / 4 for move in segment if move[2] != .0]
            dist_lin = sum(dist_lin)
            dist_arc = sum(dist_arc)
            dist = dist_lin + dist_arc

            # s = 1/2at^2
            # a - known, v = a*t, v - known
            # s = 1/2 v/t t^2 -> s = 1/2 vt
            # s = 1/2 a (v/a)^2 -> s = 1/2 v^2/a
            sa = 1/2 * vmax**2 / a  # acceleration distance
            sd = sa  # deceleration distance
            s = dist - sa - sd  # constant speed distance
            i_pts = np.linspace(dist[0], dist[-1], int(dist[-1]/(0.1*res)))  # intermediate points




def segments_from_path(path, mps):
    segments = {}
    temp_segments = []
    ang_segment = False
    while path:
        key, value = path.pop(0)
        move = mps[key][value]
        # if there is no linear dislocation (it's a turn-in-place)
        if move[0] == 0 and move[1] == 0:
            if not ang_segment:
                ang_segment = True
                segments[len(segments)] = ('linear', temp_segments)
            temp_segments = move
        else:
            # if previous move was turn-in-place
            if ang_segment:
                ang_segment = False
                segments[len(segments)] = ('angular', temp_segments)
                temp_segments = []  # reset the temp segments
            temp_segments.append(move)

    return segments