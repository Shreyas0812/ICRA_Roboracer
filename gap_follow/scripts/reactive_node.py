#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):

    def __init__(self):
        super().__init__('reactive_node')

        self.declare_parameter('window_half', 40)
        self.window_half = self.get_parameter('window_half').value

        self.declare_parameter('bubble_window', 80)
        self.bubble_window = self.get_parameter('bubble_window').value

        self.declare_parameter('speed_cap', 3.5)
        self.speed_cap = self.get_parameter('speed_cap').value

        self.declare_parameter('speed_lower_cap', 2.0)
        self.speed_lower_cap = self.get_parameter('speed_lower_cap').value

        self.declare_parameter('speed_multiplier', 1.1)
        self.speed_multiplier = self.get_parameter('speed_multiplier').value

        self.declare_parameter('alley_width', 1.2)
        self.alley_width = self.get_parameter('alley_width').value

        self.desired_dist_from_wall = self.alley_width / 2

        self.dist_threshold = self.desired_dist_from_wall * 0.7

        self.declare_parameter('door_thresh', 0.08) # 0.08
        self.door_thresh = self.get_parameter('door_thresh').value

        self.running_steering_angle = 0.0
        # gap follow parameters
        self.declare_parameter('maxActionableDist', 3.0)
        self.maxActionableDist = self.get_parameter('maxActionableDist').value


        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.lidar_sub_ = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
    
    def get_range(self, range_data, angle):
        
        index = int((angle - range_data['angles'][0]) / range_data['angle_increment'])

        start_index = index - self.window_half
        end_index = index + self.window_half

        # average of 10 points around the index 
        return_range = np.average(range_data['ranges'][start_index:end_index])

        return index, return_range

    def find_best_point(self, start_i, end_i, range_data):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        # Going to the center of the gap for now

        slow_turn1 = False
        slow_turn2 = False
        slow_turn3 = False

        running_towards_idx, running_towards_dist = self.get_range(range_data, self.running_steering_angle)

        center_point_idx = int((start_i + end_i) / 2)
        
        diff_in_index = np.abs(center_point_idx - running_towards_idx)

        if diff_in_index <= 5:
            slow_turn1 = True
        elif diff_in_index <= 10:
            slow_turn2 = True
        elif diff_in_index <= 15:
            slow_turn3 = True

        furthest_point_idx = center_point_idx

        # if running_towards_idx in range(start_i, end_i):
        #     if running_towards_idx < center_point_idx:
        #         furthest_point_idx = (running_towards_idx + center_point_idx) // 3
        #     else:
        #         furthest_point_idx = (running_towards_idx + center_point_idx) // 3
            
        #     # # slow_turn = True
        #     # furthest_point_idx = center_point_idx
                
        #     # self.get_logger().info(f'indside the gap, slow turn',throttle_duration_sec=1)
        # else:
        #     # self.get_logger().info(f'Outside the Gap, going to center', throttle_duration_sec=1)

        #     furthest_point_idx = center_point_idx

    
        return slow_turn1, slow_turn2, slow_turn3, furthest_point_idx
    
    def find_max_gap(self, free_space_ranges, gaps):
        """ Return the start index & end index of the max gap in free_space_ranges
        """

        if len(gaps) == 0:
            return (0, 1080)
        range_gap = []
        start = gaps[0]
        prev = gaps[0]

        for curr in gaps[1:]:
            if curr != prev + 1:
                range_gap.append((start, prev))
                start = curr
            prev = curr

        # Add the last range
        range_gap.append((start, prev))

        largest_range_gap = max(range_gap, key=lambda x: x[1] - x[0])

        return largest_range_gap
    
    def mutate_ranges(self, ranges, center_index, value):
        """ Mutate ranges to be a certain value
        """
        ranges[center_index-self.window_half:center_index+self.window_half] = value

        return ranges
    
    def preprocess_lidar(self, proc_ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """

        gaps = []
        for i in range(self.window_half, len(proc_ranges)-self.window_half):
            cur_mean = np.mean(proc_ranges[i-self.window_half:i+self.window_half+1])
            if cur_mean > self.maxActionableDist:
                gaps.append(i)

        for index in gaps:
            proc_ranges = self.mutate_ranges(proc_ranges, index, self.maxActionableDist)

        return proc_ranges, gaps
    
    def run_gap_follow(self, dead_straight, range_data):
        
        proc_ranges, gaps = self.preprocess_lidar(range_data['ranges'])

        min_dist_idx = np.argmin(proc_ranges)

        #Eliminate all points inside 'bubble' (set them to zero)
        bubble_start = max(0, min_dist_idx - self.bubble_window)
        bubble_end = min(len(proc_ranges), min_dist_idx + self.bubble_window)

        proc_ranges[bubble_start:bubble_end] = proc_ranges[bubble_start:bubble_end] - 1.5

        proc_ranges[proc_ranges < 0] = 0

        # Change from using arbitrary 180 to calculated values
        start_index = len(range_data['ranges']) // 6     # corresponds to -90 degrees
        end_index = (len(range_data['ranges']) * 5) // 6 # corresponds to +90 degrees
        proc_ranges[: start_index] = 0
        proc_ranges[end_index:] = 0

        gaps = list(set(gaps) - set(range(bubble_start, bubble_end+1)) - set(range(0, start_index)) - set(range(end_index, 1080)))

        start_idx, end_idx = self.find_max_gap(proc_ranges, gaps)

        slow_turn1, slow_turn2, slow_turn3, best_point = self.find_best_point(start_idx, end_idx, range_data)

        angle = range_data["angles"][0] + best_point * range_data["angle_increment"]

        if slow_turn1:
            angle = angle * 0.3
        elif slow_turn2:
            angle = angle * 0.4
        elif slow_turn3:
            angle = angle * 0.5

        speed = dead_straight * self.speed_multiplier
        if speed > self.speed_cap:
            speed = self.speed_cap
        elif speed < self.speed_lower_cap:
            speed = self.speed_lower_cap
        else:
            speed = self.speed_lower_cap

        self.publish_drive(speed, angle)

    def lidar_callback(self, msg):

        range_data = {
            'ranges': np.array(msg.ranges),
            'angles': np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges)),
            'angle_increment': msg.angle_increment
        }

        dead_straight_idx, dead_straight = self.get_range(range_data, 0)
        right_idx, right_dist = self.get_range(range_data, -np.pi/2)
        left_idx, left_dist = self.get_range(range_data, np.pi/2)
        
        total_width = left_dist + right_dist
        if total_width > self.alley_width + self.door_thresh or left_dist < self.dist_threshold or right_dist < self.dist_threshold:
            self.run_gap_follow(dead_straight, range_data)
        else:
            speed = dead_straight * self.speed_multiplier

            if speed > self.speed_cap:
                speed = self.speed_cap
            elif speed < self.speed_lower_cap:
                speed = self.speed_lower_cap
            else:
                speed = self.speed_lower_cap

            self.publish_drive(speed, 0.0)

    def publish_drive(self, speed, steering_angle): 
        # self.get_logger().info(f'publishing drive message, {speed}, {angle}', throttle_duration_sec=1)

        self.running_steering_angle = steering_angle
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub_.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()