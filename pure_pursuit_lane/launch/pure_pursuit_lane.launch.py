from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="pure_pursuit_lane",
            executable="pure_pursuit_boxes.py",
            name="pure_pursuit_boxes",
            output="screen",

        )
    ])