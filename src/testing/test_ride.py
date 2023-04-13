#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from follower import Follower
from time import time


class Rider(Node):
    def __init__(self, name='rider_node'):
        super().__init__(name)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.data = {}
        self.data["X"], self.data["Y"], self.data["Theta"], self.data["U"], self.data["T"] = self.load_data()
        
        self.create_timer(0.1, self.update_vel)
        
        self.t0 = time()  # previous time
        self.t1 = time()  # current time
        self.ts = self.data["T"].pop(0)  # get first time interval  # interval time
        self.follower = Follower(d=1, alpha=0)  # create a follower, also it has to be initialized properly (init state)
        self.x_prev = (self.data["X"][0], self.data["Y"][0], self.data["Theta"][0])

    def load_data(self, filename='trajectory.csv'):
        try:
            path = list(Path(__file__).parent.parent.glob(f"resource/{filename}"))[0]
            data = pd.read_csv(path)
        except FileNotFoundError as e:
            logging.error(e)
        except IndexError as e:
            logging.error(e)
        except Exception as e:
            logging.error(e)

        X = np.array(data["x"].to_list()).reshape(-1, 1)
        Y = np.array(data["y"].to_list()).reshape(-1, 1)
        Theta = np.array(data["0"].to_list()).reshape(-1, 1)
        U = np.hstack((np.array(data["v"].to_list()).reshape(-1, 1), np.array(data["w"].to_list()).reshape(-1, 1)))
        T = np.array(data["t"].to_list()).reshape(-1, 1)
        return list(X), list(Y), list(Theta), list(U), list(T)

    def update_vel(self):
        # todo - get data to check if there's any left
        if not self.data["T"]:
            self.send_vel([0.0, 0.0])
            return False
        pose = self.get_pos()
        x_prev = self.x_prev
        self.t1 = time()

        if self.t1 - self.t0 >= self.ts:
            # virtual lider state at the end of this time interval
            x = self.data["X"].pop(0)
            y = self.data["Y"].pop(0)
            theta = self.data["Theta"].pop(0)
            # virtual lider control at this time interval
            u = self.data["U"].pop(0)
            # next interval time
            t = self.data["T"].pop(0)

            ts = t - self.ts
            self.follower.update_pos((x, y, theta), x_prev, u)
            self.follower.update_vel(ts)
            self.send_vel(self.follower.u)

            self.ts = t  # new interval time = current time + relative time
            # self.t0 = self.get_clock().now().to_msg()  # update previous time to current time
            print(self.ts)

        return True

    def send_vel(self, u):
        # send velocity command to the '/cmd_vel' topic
        msg = Twist()
        msg.linear.x = u[0]
        msg.angular.z = u[1]
        self.pub.publish(msg)

    def get_pos(self):
        # get pose of the robot
        pass


if __name__ == '__main__':
    rclpy.init()
    rider = Rider()
    rclpy.spin(rider)
    rclpy.shutdown()
    pass
