import numpy as np
import matplotlib.pyplot as plt


class State:
    def __init__(self, x=0, y=0, yaw=0):
        self.x = x
        self.y = y
        self.yaw = yaw

    @property
    def state(self):
        return self.x, self.y, self.yaw
    

class MotionPrimitive:
    def __init__(self, **params):
        self.x = params["x"]
        self.y = params["y"]
        self.yaw = params["yaw"]
        self.cost = params["cost"]
        self.weight = params["weight"]
        self.trajectory = params["trajectory"]
        self.max_yaw = params["max_yaw"]  # max and -min difference in yaw between start and goal position in this primitive

    def __add__(self, mp):
        self.x += mp.x
        self.y += mp.y
        self.yaw += mp.yaw
        self.cost += mp.cost
        # self.weight = ? 
        # self.trajectory = ?


def move_primitive(state, mp):
    # move the state by a given motion primitve
    x = state.x + mp.x
    y = state.y + mp.y
    yaw = state.yaw + mp.yaw
    return State(x, y, yaw)


def mp_possible(state, mp):
    # is this motion primitive legal?
    return abs(state.yaw - mp.yaw) <= mp.max_yaw


def plot_states(states):
    plt.figure()
    xy = np.array([(state.x, state.y) for state in states]).reshape(-1, 2)
    plt.scatter(xy[:, 0], xy[:, 1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()


def main():
    origin = State(x=0, y=0, yaw=0)
    state_lattice = [origin]
    new_states = [origin]
    primitives = [MotionPrimitive(x=1, y=0, yaw=0, cost=0, weight=0, trajectory=None, max_yaw=30),
                  MotionPrimitive(x=0, y=1, yaw=0, cost=0, weight=0, trajectory=None, max_yaw=30),
                  MotionPrimitive(x=1, y=1, yaw=0, cost=0, weight=0, trajectory=None, max_yaw=30)]  # all possible motion primitives
    for i in range(5):  # repeat 5 times (depth = 5)
        new_states_temp = []  # states created in this depth iteration
        for state in new_states:  # for each state in newest states
            for mp in primitives:  # for each motion primitive
                if mp_possible(state, mp):  # if it's possible to use this primitive (in regards to yaw changes)
                    new_states_temp.append(move_primitive(state, mp))
            state_lattice += [state for state in new_states_temp]
        new_states = new_states_temp

    # for states in state_lattice:
    #     print(states.state)

    plot_states(state_lattice)

if __name__ == '__main__':
    main()