import math
import numpy as np
from discretized_footprint import get_conf_turn
import pandas as pd
from pathlib import Path


class Trajectory:
    def __init__(self) -> None:
        self.item = []  # [xyz], v, w, t

    def last_len(self, xyz: list) -> float:
        return math.sqrt(math.pow(self.item[-1][0][0] - xyz[0], 2) + math.pow(self.item[-1][0][1] - xyz[1], 2))
    
    def last_rot(self, xyz: list) -> float:
        return abs(self.item[-1][0][2] - xyz[2])

    def get_last_vel(self) -> float:
        return self.item[-1][1]

    def get_last_rot(self) -> float:
        return self.item[-1][2]

    def input_item(self, item : list[list, float, float, float]) -> None:
        self.item.append(item)


def traj_from_path(path, x0, path_mps, mps, vmax, wmax, a, e, res):
    # x0 - starting configuraton
    # path - list of motion primitives identifiers
    # mps - list of motion primitives
    # t - trajectory time
    # a - linear acceleration
    # e - angular acceleration
    # res - resolution
    traj = Trajectory()
    pts, inds = inter_points(x0=x0, path=path, path_mps=path_mps, mps=mps, vmax=vmax, wmax=wmax, a=a, e=e)  # points and indicators
    # the goal is to calculate velocities for given time steps and configurations

    t = .0
    vk = .0
    wk = .0
    traj.input_item((x0, vk, wk, t))
    v_decel = False
    w_decel = False
    sx = 0
    sy = 0
    st = 0
    R = 1

    decel_pt = inds.pop(0)
    for i in range(1, len(pts)):
        if (sx > decel_pt[0][0]) and (sy > decel_pt[0][1]) and decel_pt[1] and inds:   # if sx and sy greater than decel point's sx and sy, then robot should decelerate linearly
            v_decel = True
            if inds: decel_pt = inds.pop(0)
        if (st > decel_pt[0][2]) and decel_pt[2] and inds:                          # if st greater than decel point's st, then robot should decelerate angularly   
            w_decel = True
            if inds: decel_pt = inds.pop(0)

        if pts[i][0] != traj.item[-1][0][0] or pts[i][1] != traj.item[-1][0][1]:  # linear or linear and angular movement
            s = traj.last_len(pts[i])   # length of straight line movement

            if v_decel:     # linear deceleration
                sqrt = math.pow(traj.get_last_vel(), 2) - 2*a*s     # how much distance driven in this time step (if sqrt'ed)
                if sqrt > 0:
                    vk = math.sqrt(sqrt)    # new velocity
                else:
                    vk = 0
                    v_decel = False     # finished decelerating

                t += abs((vk - traj.get_last_vel()) / a)     # length of this time step

            elif (traj.get_last_vel() < vmax) and (not v_decel):    # accelerating
                vk = math.sqrt(2*a*s + math.pow(traj.get_last_vel(), 2))    # new velocity
                if vk > vmax: vk = vmax     # if new velocity greater than max velocity, cap it at max velocity
                t += abs((vk - traj.get_last_vel()) / a)   # length of this time step # t = (vk - v0) / a  WRONG ****************************  # check now

            else:   # constant speed
                vk = vmax
                t += s / vk   # length of this time step # WRONG ****************************   # check now

            if pts[i][2] != traj.item[-1][0][2]:    # there's rotation
                wk = vk / R
                wk = math.copysign(wk, pts[i][2] - traj.item[-1][0][2])
            else:
                wk = 0
            
            # update travelled distance in x and y axes
            sx += abs(traj.item[-1][0][0] - pts[i][0])
            sy += abs(traj.item[-1][0][1] - pts[i][1])
            st += abs(traj.item[-1][0][2] - pts[i][2])

            traj.input_item((pts[i], vk, wk, t))

        else:   # rotation
            # w = v/r
            s = traj.last_rot(pts[i])
            if w_decel:
                sqrt = math.pow(traj.get_last_rot(), 2) - 2*e*s     # how much distance driven in this time step (if sqrt'ed)
                if sqrt > 0:
                    wk = math.sqrt(sqrt)    # new angular velocity
                    wk = math.copysign(wk, pts[i][2] - traj.item[-1][0][2])
                else:
                    wk = 0
                    w_decel = False     # finished decelerating

                t += abs((wk - traj.get_last_rot()) / e)     # length of this time step

            elif (abs(traj.get_last_rot()) < wmax) and (not w_decel):    # accelerating
                wk = math.sqrt(2*e*s + math.pow(traj.get_last_rot(), 2))    # new velocity
                if wk > wmax: wk = wmax     # if new velocity greater than max velocity, cap it at max velocity
                wk = math.copysign(wk, pts[i][2] - traj.item[-1][0][2])
                t += abs((wk - traj.get_last_rot()) / e)   # length of this time step # t = (vk - v0) / a  WRONG ****************************  # check now
                
            else:   # constant speed
                t += s / wmax   # length of this time step # WRONG ****************************   # check now
                wk = wmax
                wk = math.copysign(wk, pts[i][2] - traj.item[-1][0][2])

            traj.input_item((pts[i], 0, wk, t))
            st += s

    # trajectory = [pt for pt in traj.item]
    # for pt in trajectory:
    #     print(pt)
    traj_to_csv(traj.item)
    return traj.item


def traj_to_csv(traj, filename='trajectory.csv'):
    # save results to .csv
    x = np.array([x for x, v, w, t in traj])
    u = np.array([[v, w] for x, v, w, t in traj])
    t = np.array([t for x, v, w, t in traj]).reshape(-1, 1)
    results = pd.DataFrame(np.hstack((x, u, t)))
    results.columns = ["x", "y", "0", "v", "w", "t"]
    path = list(Path(__file__).parent.parent.parent.glob(f"testing/anytime_d_star/{filename}"))[0]
    results.to_csv(path, index=False)


def inter_points(x0, path, path_mps, mps, vmax, wmax, a, e) -> tuple:
    # return intermediate points between node points : states
    # and points where the section begins or ends : seg_ind
    segments = segments_from_path(path=path, path_mps=path_mps, mps=mps)
    num_ang_inte = 10
    num_lin_inte = 50
    # travelled_distance = 0
    prev_state = x0
    states = []
    seg_ind = []  # segment start and finish indicators
    # s = 1/2at^2
    # a - known, v = a*t, v - known
    # s = 1/2 v/t t^2 -> s = 1/2 vt
    # s = 1/2 a (v/a)^2 -> s = 1/2 v^2/a
    sa = 1/2 * vmax**2 / a  # linear acceleration and deceleration distance
    se = 1/2 * wmax**2 / e  # angular acceleration and deceleration distance

    sx = 0
    sy = 0
    st = 0

    # add all points that reach vmax/wmax when accelerating and when starting to decelerate 
    for type, segment in segments.values():

        if type == 'angular':  # only for 90 degree turns, 45 degree turns are better approximated by a diagonal

            for i in range(len(segment)):
                ori = segment[i][2]  # if 0 <= segment[i][2] <= math.pi else -(2*math.pi - segment[i][2])
                intr_ori = np.linspace(prev_state[2], ori, num=num_ang_inte)    # intermediate orientations
                x = segment[i][0]; y = segment[i][1]    # new x and y
                intr_states = [(x, y, theta) for theta in intr_ori]     # intermediate states (x, y, theta)
                states += intr_states[:-1]

                prev_state = segment[i]  # update previous state to current state

            st += abs(segment[0][2] - segment[-1][2])
            seg_ind.append([[sx, sy, st-se], None, wmax])  # end of this segment deceleration

        else:
            for i in range(len(segment)):
                ori_dif = segment[i][2] - prev_state[2]  # will probably not work straigh away due to 2pi wrap
                # ori_dif = (ori_dif % -math.pi) if ori_dif >= math.pi else ori_dif % math.pi
                # ori_dif = ori_dif % math.pi
                # if ori_dif >
                if ori_dif == 0:    #  straight movement, no rotation
                    intr_lin_x = np.linspace(prev_state[0], segment[i][0], num=num_lin_inte)
                    intr_lin_y = np.linspace(prev_state[1], segment[i][1], num=num_lin_inte)
                    theta = segment[i][2]
                    intr_states = [(x, y, theta) for x, y in zip(intr_lin_x, intr_lin_y)]
                    states += intr_states[:-1]

                    sx += abs(prev_state[0] - segment[i][0])
                    sy += abs(prev_state[1] - segment[i][1])

                    prev_state = segment[i]

                else:   # turn (linear and angular speed)
                    new_state = segment[i]
                    # ori_dif = new_state[2] - prev_state[2]  # will probably not work straigh away due to 2pi wrap
                    radius = new_state[0] - prev_state[0]
                    ori_dif = ori_dif if ori_dif <= math.pi else ori_dif % (-math.pi)
                    dir = "cw" if math.copysign(1, ori_dif) == -1 else "ccw"  # if sign positive -> move ccw, else cw
                    intr_states = get_conf_turn(abs(radius), n_samples=num_lin_inte, start_angle=prev_state[2], stop_angle=new_state[2], direction=dir)
                    x0 = prev_state[0]; y0 = prev_state[1]
                    intr_states = [(x+x0, y+y0, theta%(math.pi)) for x, y, theta in intr_states]
                    states += intr_states[:-1]

                    # find the point where the turn ends
                    k = 0
                    for j in range(i, len(segment)):
                        if j == len(segment)-1:
                            k = j
                            break
                        if segment[j][2] - segment[j+1][2] == 0:
                            k = j
                            break

                    st += abs(segment[k][2] - prev_state[2])
                    seg_ind.append([[sx, sy, st-se], None, wmax])
                    sx += abs(radius)
                    sy += abs(radius)

                    prev_state = segment[i]

            seg_ind.append([[sx-sa, sy-sa, st], vmax, None])

    seg_ind.append([[sx-sa, sy-sa, st-se], vmax, wmax])
    print(states)
    print(seg_ind)
    
    return states, seg_ind


def segments_from_path(path, path_mps, mps):
    segments = {}
    temp_segments = []
    ang_segment = False
    path.pop(0)
    while path_mps:
        key, value = path_mps.pop(0)
        xy = path.pop(0)
        ori = xy[2] if 0 <= xy[2] <= math.pi else -(2*math.pi - xy[2])
        xy = [xy[0], xy[1], ori]
        move = mps[key][value]
        # if there is no linear dislocation (it's a turn-in-place)
        if move[0] == 0 and move[1] == 0:
            if not ang_segment:
                ang_segment = True  # now an angular segment
                segments[len(segments)] = ('linear', temp_segments)  # add new segment of type 'linear' to the dictionary
                temp_segments = [xy]  # set temporary segments to the current move
            else:
                temp_segments.append(xy)
        else:
            # if previous move was turn-in-place
            if ang_segment:
                ang_segment = False  # no more angular segment
                segments[len(segments)] = ('angular', temp_segments)  # add new segment of type 'angular' to the dictionary
                temp_segments = []  # reset the temp segments
            temp_segments.append(xy)  # add current move to temporary segment list

    segments[len(segments)] = ('linear', temp_segments)

    return segments