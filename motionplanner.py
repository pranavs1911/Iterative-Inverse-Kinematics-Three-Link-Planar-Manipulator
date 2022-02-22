"""
motionplanner.py
"""

import time
from typing import List, Tuple, Union

import numpy as np
import pygame
from decimal import Decimal


class Robot:

    """
    Defining maximum velocity, maximum acceleration, joint limits before
    implementing the motion planner.
    """
    joint_l = [-6.28, 6.28]
    velm = 50
    accm = 75
    dt = 0.033

    link_1: float = 65.  # pixels
    link_2: float = 40.  # pixels
    link_3: float = 30.
    _theta_0: float  # radians
    _theta_1: float  # radians
    _theta_2: float  # radians

    def __init__(self) -> None:
        # internal variables
        self.t0_aggr: List[float] = []
        self.t1_aggr: List[float] = []
        self.t2_aggr: List[float] = []

        self.theta_0 = 0.
        self.theta_1 = 0.
        self.theta_2 = 0.

    # Getters/Setters
    @property
    def theta_0(self) -> float:
        return self._theta_0

    @theta_0.setter
    def theta_0(self, value: float) -> None:
        self.t0_aggr.append(value)
        self._theta_0 = value
        # Check limits
        assert self.angle_limits(value), \
            f'Joint 0 value {value} exceeds joint limits'
        assert self.vel_max(self.t0_aggr) < self.velm, \
            f'Joint 0 Velocity {self.vel_max(self.t0_aggr)} exceeds velocity limit'
        assert self.acc_max(self.t0_aggr) < self.accm, \
            f'Joint 0 Accel {self.acc_max(self.t0_aggr)} exceeds acceleration limit'

    @property
    def theta_1(self) -> float:
        return self._theta_1

    @theta_1.setter
    def theta_1(self, value: float) -> None:
        self.t1_aggr.append(value)
        self._theta_1 = value
        assert self.angle_limits(value), \
            f'Joint 1 value {value} exceeds joint limits'
        assert self.vel_max(self.t1_aggr) < self.velm, \
            f'Joint 1 Velocity {self.vel_max(self.t1_aggr)} exceeds velocity limit'
        assert self.acc_max(self.t1_aggr) < self.accm, \
            f'Joint 1 Accel {self.acc_max(self.t1_aggr)} exceeds acceleration limit'

    @property
    def theta_2(self) -> float:
        return self._theta_2

    @theta_2.setter
    def theta_2(self, value: float) -> None:
        self.t2_aggr.append(value)
        self._theta_2 = value
        assert self.angle_limits(value), \
            f'Joint 2 value {value} exceeds joint limits'
        assert self.vel_max(self.t2_aggr) < self.velm, \
            f'Joint 2 Velocity {self.vel_max(self.t2_aggr)} exceeds velocity limit'
        assert self.acc_max(self.t2_aggr) < self.accm, \
            f'Joint 2 Accel {self.acc_max(self.t2_aggr)} exceeds acceleration limit'

    # Kinematics
    def pos_end1(self) -> Tuple[float, float]:
        """
        Joint 1 Position
        """
        return self.link_1 * np.cos(self.theta_0), self.link_1 * np.sin(self.theta_0)

    def pos_end2(self) -> Tuple[float, float]:
        """
        Joint 2 Position
        """
        return self.link_1 * np.cos(self.theta_0) + self.link_2 * np.cos(
            self.theta_0 + self.theta_1), self.link_1 * np.sin(self.theta_0) + self.link_2 * np.sin(
            self.theta_0 + self.theta_1)

    def pos_end3(self) -> Tuple[float, float]:
        """
        Joint 3 Position
        """
        return self.forward(self.theta_0, self.theta_1, self.theta_2)

    @classmethod
    def angle_limits(cls, theta: float) -> bool:
        return cls.joint_l[0] < theta < cls.joint_l[1]

    @classmethod
    def vel_max(cls, all_theta: List[float]) -> float:
        return float(abs(max(np.diff(all_theta) / cls.dt, default=0.)))

    @classmethod
    def acc_max(cls, all_theta: List[float]) -> float:
        return float(abs(max(np.diff(np.diff(all_theta)) / cls.dt / cls.dt, default=0.)))

    @classmethod
    def radius_shortest(cls) -> float:
        return max((cls.link_1 - (cls.link_2 + cls.link_3)), 0)

    @classmethod
    def radius_longest(cls) -> float:
        return cls.link_1 + cls.link_2 + cls.link_3

    @classmethod
    def dirkin(cls, theta_0: float, theta_1: float, theta_2: float) -> Tuple[float, float]:
        """
        Compute the x, y position of the end of the links from the joint angles
        """
        x = cls.link_1 * np.cos(theta_0) + cls.link_2 * np.cos(theta_0 + theta_1) + cls.link_3 * np.cos(
            theta_0 + theta_1 + theta_2)
        y = cls.link_1 * np.sin(theta_0) + cls.link_2 * np.sin(theta_0 + theta_1) + cls.link_3 * np.sin(
            theta_0 + theta_1 + theta_2)

        return x, y
    @classmethod

    #Intersection of two lines has been referenced from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/amp/


    def Point(cls, x, y) -> Tuple [float, float]:
        cls.x = x
        cls.y = y
        return x,y
    # Given three colinear points p, q, r, the function checks if
    # point q lies on line segment 'pr'

    def onSegment( cls, p:Tuple[float,float], q:Tuple[float,float], r:Tuple[float,float]) -> bool:
        if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
                (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False

    def orientation(cls, p:Tuple[float,float], q:Tuple[float,float], r:Tuple[float,float]) -> int:
        # to find the orientation of an ordered triplet (p,q,r)
        # function returns the following values:
        # 0 : Colinear points
        # 1 : Clockwise points
        # 2 : Counterclockwise
        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/amp/
        # for details of below formula.
        p = cls.Point(p[0],p[1])
        q = cls.Point(q[0],q[1])
        r = cls.Point(r[0],r[1])
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if (val > 0):
            # Clockwise orientation
            return 1
        elif (val < 0):
            # Counterclockwise orientation
            return 2
        else:
            # Colinear orientation
            return 0
    # The main function that returns true if
    # the line segment 'p1q1' and 'p2q2' intersect.

    def doIntersect(cls, p1:Tuple[float, float], q1:Tuple[float,float], p2:Tuple[float, float], q2:Tuple[float,float]) -> bool:
        # Find the 4 orientations required for
        # the general and special cases
        o1 = cls.orientation(cls,p1, q1, p2)
        o2 = cls.orientation(cls,p1, q1, q2)
        o3 = cls.orientation(cls, p2, q2, p1)
        o4 = cls.orientation(cls, p2, q2, q1)
        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True
        # Special Cases
        # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
        if ((o1 == 0) and cls.onSegment(cls, p1, p2, q1)):
            return True
        # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
        if ((o2 == 0) and cls.onSegment(cls, p1, q2, q1)):
            return True
        # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
        if ((o3 == 0) and cls.onSegment(cls, p2, p1, q2)):
            return True
        # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
        if ((o4 == 0) and cls.onSegment(cls, p2, q1, q2)):
            return True
        # If none of the cases
        return False

    @classmethod
    ##Referenced inverse kinematics from https://www.youtube.com/watch?v=sJudoHiWes8

    def invkin(cls, x: float, y: float) -> Tuple[float, float, float]:
        """
        Compute the joint angles from the position of the end of the links
        """
        print("X",x)
        print("Y",y)
        theta_0 = np.random.randint(-6.28, 6.28)
        theta_1 = np.random.randint(-6.28, 6.28)
        theta_2 = np.random.randint(-6.28, 6.28)
        inc: float = 0.1
        k: float = 0.01
        while (inc >= k):
            """
                Runs the iteration for different steps (0.1, 0.01, 0.001)
            """
            phi = 18.84
            phi = format(phi, '.9f')
            phi = float(phi)
            while (phi >= -18.84):
                """
                    Runs the loop as long as phi stays between 18.84 and -18.84
                """
                phi = format(phi, '.9f')
                phi = float(phi)
                u = x - cls.link_3 * np.cos(phi)
                v = y - cls.link_3 * np.sin(phi)
                calcgcostheta1 = (u** 2 + v**2 - cls.link_1** 2 - cls.link_2**2) / (2 * cls.link_1 * cls.link_2)
                if ((-1 <= calcgcostheta1 <= 1)):
                    """
                        Calculates value of theta1 if cos(theta1) lies between -1 and +1
                    """
                    calcsintheta1 = np.sqrt(1 - calcgcostheta1**2)
                    calcsintheta1dup = -np.sqrt(1 - calcgcostheta1**2)
                    theta_1_dup1 = np.arctan2(calcsintheta1, calcgcostheta1)
                    theta_1_dup2 = np.arctan2(calcsintheta1dup, calcgcostheta1)
                    del1 = (cls.link_1 + cls.link_2 * np.cos(theta_1_dup1))** 2 + ((cls.link_2**2) * (np.sin(theta_1_dup1))** 2)
                    deldup = (cls.link_1 + cls.link_2 * np.cos(theta_1_dup2))** 2 + ((cls.link_2** 2) * (np.sin(theta_1_dup2))** 2)
                    calcsintheta0 = (v * (cls.link_1 + cls.link_2 * np.cos(theta_1_dup1)) - u * (cls.link_2 * np.sin(theta_1_dup1))) / del1
                    calccostheta0 = (u * (cls.link_1 + cls.link_2 * np.cos(theta_1_dup1)) - v * (cls.link_2 * np.sin(theta_1_dup1))) / del1
                    calcsintheta0dup = (v * (cls.link_1 + cls.link_2 * np.cos(theta_1_dup2)) - u * (cls.link_2 * np.sin(theta_1_dup2))) / deldup
                    calccostheta0dup = (u * (cls.link_1 + cls.link_2 * np.cos(theta_1_dup2)) - v * (cls.link_2 * np.sin(theta_1_dup2))) / deldup

                    if ((-1 <= calcsintheta0 <= 1 and -1 <= calccostheta0 <= 1) or (-1 <= calcsintheta0dup <= 1 and -1 <= calccostheta0dup <= 1)):
                        """
                            Calculates 2 values of theta0 if cos(theta0) lies between -1 and +1 for both values.
                        """
                        if (-1 <= calcsintheta0 <= 1 and -1 <= calccostheta0 <= 1 ):
                            """
                                Calculates 1 value of theta0 if cos(theta0) lies between -1 and +1
                            """
                            theta_0_dup1 = np.arctan2(calcsintheta0, calccostheta0)
                            if (cls.angle_limits(theta_0_dup1)):
                                """
                                    Calculates value of theta2 if theta0 lies between -6.28 and +6.28
                                """
                                theta_2_dup1 = phi - (theta_0_dup1 + theta_1_dup1)
                                if (cls.angle_limits(theta_2_dup1)):
                                    """
                                        Checks angle limit of theta_2_dup1 and calculates position of points p1, p2 , p0 and p3.
                                    """
                                    Xdup1, Ydup1 = cls.dirkin(theta_0_dup1, theta_1_dup1, theta_2_dup1)
                                    p3 = cls.Point(Xdup1, Ydup1)
                                    p2 = cls.Point(cls.link_1*np.cos(theta_0_dup1) + cls.link_2*np.cos(theta_0_dup1 + theta_1_dup1), cls.link_1*np.sin(theta_0_dup1) + cls.link_2*np.sin(theta_0_dup1 + theta_1_dup1))
                                    p1 = cls.Point(cls.link_1*np.cos(theta_0_dup1), cls.link_1*np.sin(theta_0_dup1))
                                    p0: Tuple[float, float] = (0,0)
                                    #print(p2)
                                    if (np.sqrt((x - Xdup1) ** 2 + (y - Ydup1) ** 2) <= 0.25):

                                        if cls.doIntersect( cls,p2, p3, p0, p1) == 0 and cls.doIntersect(cls, p0, p1, p1, p2) == 1 and cls.doIntersect(cls, p1, p2, p2, p3) == 1:
                                            """
                                            Finalizes first set of theta_0, theta_1 and theta_2 if end point is within acceptable range. 
                                            """
                                            theta_0 = theta_0_dup1
                                            theta_1 = theta_1_dup1
                                            theta_2 = theta_2_dup1
                                            break

                        if (-1 <= calcsintheta0dup <= 1 and -1 <= calccostheta0dup <= 1 ):
                            """
                            Calculates theta_0, theta_2
                            """
                            theta_0_dup2 = np.arctan2(calcsintheta0dup, calccostheta0dup)
                            if (cls.angle_limits(theta_0_dup2)):
                               # print("enter2")
                                theta_2_dup2 = phi - (theta_0_dup2 + theta_1_dup2)
                                if (cls.angle_limits(theta_2_dup2)):
                                    """
                                    Checks angle limit of theta_2_dup1 and calculates position of points p1, p2 , p0 and p3.
                                    """
                                    Xdup2, Ydup2 = cls.dirkin(theta_0_dup2, theta_1_dup2, theta_2_dup2)
                                    p3 = cls.Point(Xdup2, Ydup2)
                                    p2 = cls.Point(cls.link_1 * np.cos(theta_0_dup2) + cls.link_2 * np.cos(theta_0_dup2 + theta_1_dup2),cls.link_1 * np.sin(theta_0_dup2) + cls.link_2 * np.sin(theta_0_dup2 + theta_1_dup2))
                                    p1 = cls.Point(cls.link_1 * np.cos(theta_0_dup2),cls.link_1 * np.sin(theta_0_dup2))

                                    p0 :Tuple[float, float] = (0,0)

                                    if (np.sqrt((x - Xdup2) ** 2 + (y - Ydup2) ** 2) <= 0.25):
                                        if cls.doIntersect(cls,p0, p1, p2, p3) == 0 and cls.doIntersect(cls, p0, p1, p1, p2) == 1 and cls.doIntersect(cls, p1, p2, p2, p3) == 1:
                                            """
                                            Finalizes second set of theta_0, theta_1 and theta_2 if end point is within acceptable range. 
                                            """
                                            theta_0 = theta_0_dup2
                                            theta_1 = theta_1_dup2
                                            theta_2 = theta_2_dup2
                                            break
                phi = phi - inc
                phi = format(phi, '.9f')
                phi = float(phi)
            Xdup1, Ydup1 = cls.dirkin(theta_0_dup1, theta_1_dup1, theta_2_dup1)
            Xdup2, Ydup2 = cls.dirkin(theta_0_dup2, theta_1_dup2, theta_2_dup2)
            #print(Xdup2, Ydup2)
            if ((np.sqrt((x - Xdup1) ** 2 + (y - Ydup1) ** 2) <= 0.25) or (np.sqrt((x - Xdup2) ** 2 + (y - Ydup2) ** 2) <= 0.25)):
                break
            """
            Decreases increment angle by 0.1 if no solution is found for phi between 18.84 and -18.84
            """
            inc = inc/10
            inc = format(inc, '.9f')
            inc = float(inc)
            print(inc)
            if (inc<k):
                k = k/10
                k = format(k, '.9f')
                k = float(k)
            #print(inc)
        return theta_0, theta_1, theta_2


def end_point(min_radius: float, max_radius: float) -> Tuple[int, int]:
    """
    Generate a random goal that is reachable by the robot arm
    """
    # Ensure theta is not 0
    theta = (np.random.random() + np.finfo(float).eps) * 2 * np.pi
    # Ensure point is reachable
    r = np.random.uniform(low=min_radius, high=max_radius)

    x = int(r * np.cos(theta))
    y = int(r * np.sin(theta))
    #x = -53
    #y = -84
    return x, y


def main() -> None:
    height = 400
    width = 400

    robot_origin = (int(width / 2), int(height / 2))
    goal = end_point(Robot.radius_shortest(), Robot.radius_longest())

    robot = Robot()
    inv = robot.invkin(goal[0],goal[1])
    print("Solution found: Theta 0: ",inv[0],"Theta 1: ",inv[1], "Theta 2: ",inv[2])



if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        print(f'ERROR: {e}, Aborting.')
    except KeyboardInterrupt:
        pass

