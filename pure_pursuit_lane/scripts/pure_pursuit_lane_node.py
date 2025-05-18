#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry

from transforms3d.euler import quat2euler
import numpy as np

from ament_index_python.packages import get_package_share_directory

class PurePursuitLaneNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit_lane_node")
        self.get_logger().info("Hello from Pure Pursuit Lane Node")

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitLaneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()