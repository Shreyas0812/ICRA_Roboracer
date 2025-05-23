#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from scipy.spatial import KDTree, transform
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from shapely.geometry import Point, Polygon


from geometry_msgs.msg import PointStamped


class PurePursuit(Node):
    """ 
    Implement Pure Pursuit + rectangular speed & lookahead zones + Lidar-based braking.
    Also visualizes:
      - Waypoints and box outlines
      - The lookahead circle around the car
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.sim = True

        # Subscribe to odometry or pose
        if self.sim:
            odom_topic = "/ego_racecar/odom"
            self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        else:
            odom_topic = "/pf/viz/inferred_pose"
            self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)

        # Publisher for drive commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Subscribe to Lidar scan for dynamic braking
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.latest_scan = None

        self.waypoint_pub = self.create_publisher(PointStamped, '/pure_pursuit/goal_waypoint', 10)

        # Default Pure Pursuit parameters
        # We'll now treat this as the "default" or fallback L if no zone overrides it
        self.default_L = 0.8 #2.0
        self.default_kp = 0.435
        self.default_speed = 1.5

        self.kv = 0.0 

        self.prev_speed = 0.0
        self.prev_steering_angle = 0.0


        self.run_gap_follow = False


        # Load waypoints and build KD-tree
        csv_data = np.loadtxt(
            "./final_points.csv",
            delimiter=",",
            skiprows=0
        )
        self.waypoints = csv_data[:, 0:2]
        self.kd_tree = KDTree(self.waypoints)

        csv_data2 = np.loadtxt(
            "./final_points_left.csv",
            delimiter=",",
            skiprows=0
        )
        self.waypoints2 = csv_data2[:, 0:2]
        self.kd_tree2 = KDTree(self.waypoints2)

        # Track which lane we're following (primary=1, alternative=2)
        self.current_waypoints = 1
        self.in_overtake_zone = False
        
        # -- BOX DEFINITIONS --
        # Now each zone includes both "speed" and "lookahead" so we can override speed and L:
        self.speed_zones = [
            {
                "name": "Box1",
                "corners": [(15.40, -10.13), (14.46, -11.99), (1.75, -8.77), (2.48, -6.25)],
                "speed": 8.0,
                "lookahead": 3.0,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True  # Enable overtaking in this box
            },
            {
                "name": "Box2",
                "corners": [ (2.48, -6.25), (1.75, -8.77),  (-6.44, -6.32), (-6.18, -4.58)],
                "speed": 1.0,
                "lookahead": 3.0,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True
            },
            {
                "name": "Box3",
                "corners": [(-2.54, -5.21),(-6.18, -4.58), (-5.8, -1.0), (-2.34, -1.23)],
                "speed": 3.5,
                "lookahead": 3.0,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True
            },
            {
                "name": "Box4",
                "corners": [(-5.8, -1.0), (-5.87, 1.74), (7.63, 0.47), (7.40, -1.22)],
                "speed": 6.0,
                "lookahead": 2.7,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True
            },
            {
                "name": "Box5",
                "corners": [(7.40, -1.22), (7.63, 0.47), (11.45, -0.26), (11.31, -2.1)],
                "speed": 2.5,
                "lookahead": 2.0,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True
            },
            {
                "name": "Box6",
                "corners": [(12.94, -3.89),(11.45, -0.26), (21.80,-2.13), (21.26, -6.57)],
                "speed": 2.5,
                "lookahead": 1.5,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True
            },
            {
                "name": "Box7",
                "corners": [(21.26, -6.57), (20.11, -8.61), (10.37, -5.17), (11.18, -3.16)],
                "speed": 4.5,
                "lookahead": 2.0,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True  # Enable overtaking in this box
            },
            {
                "name": "Box8",
                "corners": [(11.63, -2.19), (10.37, -5.17), (2.35, -3.15), (2.66, -1.03)],
                "speed": 5.5,
                "lookahead": 3.0,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True
            },
            {
                "name": "Box9",
                "corners": [(2.66, -1.03), (-2.11, -1.27), (-1.81, -4.96), (2.35, -3.15)],
                "speed": 2.5,
                "lookahead": 2.5,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True  # Enable overtaking in this box
            },
            {
                "name": "Box10",
                "corners": [(-1.81, -4.96), (0.39, -3.96), (6.80, -4.30), (6.42, -6.5)],
                "speed": 5.0,
                "lookahead": 2.7,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True  # Enable overtaking in this box
            },
            {
                "name": "Box11",
                "corners": [(6.80, -4.30), (6.42, -6.5), (10.07, -8.60), (11.54, -5.91)],
                "speed": 3.0,
                "lookahead": 1.5,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True  # Enable overtaking in this box
            },
            {
                "name": "Box12",
                "corners": [(10.07, -8.60), (11.54, -5.91), (14.68, -6.69), (13.86,-9.44)],
                "speed": 5.5,
                "lookahead": 3.5,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True  # Enable overtaking in this box
            },
            {
                "name": "Box13",
                "corners": [(14.68, -6.69), (13.86,-9.44),(18.93, -11.24), (19.75, -8.58)],
                "speed": 2.0,
                "lookahead": 1.0,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True  # Enable overtaking in this box
            },
            {
                "name": "Box14",
                "corners": [(15.40, -10.13), (14.46, -11.99),(19.62, -12.03), (18.93, -11.24)],
                "speed": 2.5,
                "lookahead": 2.0,
                "kp": 1.0,
                "kv" : 0.0,
                "overtake" : True  # Enable overtaking in this box
            },
        ]

        # Create marker publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints_markers', 10)
        self.lookahead_marker_pub = self.create_publisher(Marker, '/lookahead_marker', 10)
        self.active_path_marker_pub = self.create_publisher(Marker, '/active_path_marker', 10)

        # Timer to periodically publish markers (waypoints + boxes)
        self.timer = self.create_timer(1.0, self.publish_markers)
        self.last_switch_time = self.get_clock().now()  # Initialization
        self.switch_cooldown =  1.4


        #GAP FOLLOW PARAMETERS
        ##########################################################################################################################

        self.declare_parameter('window_half', 40)
        self.window_half = self.get_parameter('window_half').value

        self.declare_parameter('bubble_window', 80)
        self.bubble_window = self.get_parameter('bubble_window').value

        self.declare_parameter('speed_cap', 3.5)
        self.speed_cap = self.get_parameter('speed_cap').value

        self.declare_parameter('speed_lower_cap', 0.5)
        self.speed_lower_cap = self.get_parameter('speed_lower_cap').value

        self.declare_parameter('speed_multiplier_g', 1.1)
        self.speed_multiplier_g = self.get_parameter('speed_multiplier_g').value

        self.declare_parameter('alley_width', 1.2)
        self.alley_width = self.get_parameter('alley_width').value

        self.desired_dist_from_wall = self.alley_width / 2

        self.dist_threshold = self.desired_dist_from_wall * 0.7

        self.declare_parameter('door_thresh', 0.08) # 0.08
        self.door_thresh = self.get_parameter('door_thresh').value

        self.running_steering_angle = 0.0
        # gap follow parameters
        self.declare_parameter('maxActionableDist', 2.0)
        self.maxActionableDist = self.get_parameter('maxActionableDist').value

        ##########################################################################################################################

    ##########################################################################################################################

    def get_range(self, range_data, angle):
        
        index = int((angle - range_data['angles'][0]) / range_data['angle_increment'])

        start_index = index - self.window_half
        end_index = index + self.window_half

        # average of 10 points around the index 
        return_range = np.average(range_data['ranges'][start_index:end_index])

        return index, return_range

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

        return slow_turn1, slow_turn2, slow_turn3, furthest_point_idx

    def run_gap_follow_func(self, dead_straight, range_data):
        
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

        speed = dead_straight * self.speed_multiplier_g
        if speed > self.speed_cap:
            speed = self.speed_cap
        elif speed < self.speed_lower_cap:
            speed = self.speed_lower_cap
        else:
            speed = self.speed_lower_cap

        self.publish_drive_gap_follow(speed, angle)

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
    
    def scan_callback(self, scan_msg):
        """Save the latest Lidar scan for dynamic braking logic."""
        self.latest_scan = scan_msg

        if not self.run_gap_follow:
            return

        range_data = {
            'ranges': np.array(scan_msg.ranges),
            'angles': np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges)),
            'angle_increment': scan_msg.angle_increment
        }

        dead_straight_idx, dead_straight = self.get_range(range_data, 0)
        right_idx, right_dist = self.get_range(range_data, -np.pi/2)
        left_idx, left_dist = self.get_range(range_data, np.pi/2)

        current_time = self.get_clock().now()
        if not hasattr(self, 'last_gap_follow_time'):
            self.last_gap_follow_time = current_time
            self.run_gap_follow_func(dead_straight, range_data)
        else:
            elapsed = (current_time - self.last_gap_follow_time).nanoseconds / 1e9
            if elapsed >= 0.5:
                self.last_gap_follow_time = current_time
                self.run_gap_follow_func(dead_straight, range_data)
            else:
                self.get_logger().info(f"Time elapsed since last gap follow: {elapsed:.2f}s", throttle_duration_sec=0.1)


    def publish_drive_gap_follow(self, speed, steering_angle):
        self.running_steering_angle = steering_angle
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    ##########################################################################################################################

    def publish_markers(self):
        """
        Publish a MarkerArray containing:
          1) A small sphere for each waypoint.
          2) A line strip for each box corner set.
        """
        marker_array = MarkerArray()
        marker_id = 0

        # --- Waypoint Markers ---
        # Primary waypoints in red
        for i, wp in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints_primary"
            marker.id = marker_id
            marker_id += 1

            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wp[0]
            marker.pose.position.y = wp[1]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color.a = 1.0 
            marker.color.r = 1.0 
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
            
        # Alternate waypoints in blue
        for i, wp in enumerate(self.waypoints2):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints_alternate"
            marker.id = marker_id
            marker_id += 1

            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wp[0]
            marker.pose.position.y = wp[1]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color.a = 1.0 
            marker.color.r = 0.0 
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)

        # --- Box Markers (Line Strips) ---
        for box in self.speed_zones:
            corners = box["corners"]
            line_strip = Marker()
            line_strip.header.frame_id = "map"
            line_strip.header.stamp = self.get_clock().now().to_msg()
            line_strip.ns = "boxes"
            line_strip.id = marker_id
            marker_id += 1

            line_strip.type = Marker.LINE_STRIP
            line_strip.action = Marker.ADD
            line_strip.scale.x = 0.05  # thickness of lines
            line_strip.color.a = 1.0
            
            # Color boxes differently based on overtake property
            if box["overtake"]:
                line_strip.color.r = 0.0
                line_strip.color.g = 1.0  # green for overtaking zones
                line_strip.color.b = 1.0  # with blue tint
            else:
                line_strip.color.r = 0.0
                line_strip.color.g = 1.0  # green for regular zones
                line_strip.color.b = 0.0

            # Close the loop by repeating the first corner at the end
            for corner in corners:
                pt = self.create_point(corner[0], corner[1], 0.0)
                line_strip.points.append(pt)
            first_corner = corners[0]
            line_strip.points.append(self.create_point(first_corner[0], first_corner[1], 0.0))

            marker_array.markers.append(line_strip)

        # Publish all markers
        self.marker_pub.publish(marker_array)
        
        # Publish which path we're currently following
        self.publish_active_path_marker()

    def publish_active_path_marker(self):
        """
        Publish a text marker showing which path we're currently following
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "active_path"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position the text slightly above the map
        marker.pose.position.x = -10.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 2.0
        
        marker.scale.z = 1.0  # Text size
        
        if self.current_waypoints == 1:
            marker.text = "Following: PRIMARY PATH"
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:
            marker.text = "Following: ALTERNATE PATH"
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            
        marker.color.a = 1.0
        
        self.active_path_marker_pub.publish(marker)

    def pose_callback(self, pose_msg):
        # 1) Extract car position/orientation
        if self.sim:
            car_x = pose_msg.pose.pose.position.x
            car_y = pose_msg.pose.pose.position.y
            quat = pose_msg.pose.pose.orientation
        else:
            car_x = pose_msg.pose.position.x
            car_y = pose_msg.pose.position.y
            quat = pose_msg.pose.orientation
        
        quat = [quat.x, quat.y, quat.z, quat.w]
        R = transform.Rotation.from_quat(quat)
        self.rot = R.as_matrix()

        # 2) Determine current zone parameters
        speed = self.default_speed
        L = self.default_L
        kp = self.default_kp
        kv = 0.0
        self.in_overtake_zone = True
        current_box = None
        
        for box in self.speed_zones:
            if self.is_point_in_box(car_x, car_y, box["corners"]):
                speed = box["speed"]
                L = box["lookahead"]
                kv = box["kv"]
                kp = box["kp"]
                self.in_overtake_zone = box["overtake"]
                current_box = box["name"]
                self.get_logger().info(f"IN {current_box} - speed: {speed:.2f} | L: {L:.2f} | kp: {kp:.2f} | kv: {kv:.2f}", throttle_duration_sec=1.0)
                break

        # 3) Check for active cooldown period
        current_time = self.get_clock().now()
        if hasattr(self, 'switch_start_time'):
            elapsed_time = (current_time - self.switch_start_time).nanoseconds / 1e9
            if elapsed_time < self.switch_cooldown:
                L = 2.0  # Force 2m lookahead during cooldown
                # self.get_logger().info(f"Cooldown active: {self.switch_cooldown - elapsed_time:.1f}s remaining")

        # 4) Obstacle detection and lane switching logic
        min_forward_dist = self.get_min_forward_distance()
        
        # if self.in_overtake_zone:
        #     # Switch to alternate lane if obstacle detected
        #     if self.current_waypoints == 1 and min_forward_dist < 3.0:
        #         self.current_waypoints = 2
        #         self.switch_start_time = current_time
        #         L = 0.5  # Immediate lookahead override
        #         self.get_logger().info(
        #             f"SWITCHED TO ALTERNATE in {current_box} - "
        #             f"Obstacle at {min_forward_dist:.2f}m | {self.switch_cooldown}s cooldown"
        #         )
            
        #     # Check alternate lane conditions
        #     elif self.current_waypoints == 2:
        #         # Emergency switch back if alternate path blocked
        #         if min_forward_dist < 2.0:
        #             self.current_waypoints = 1
        #             self.switch_start_time = current_time  # Reset cooldown timer
        #             L = 1.0  # Maintain 2m lookahead
        #             self.get_logger().warning(
        #                 f"EMERGENCY SWITCH BACK in {current_box} - "
        #                 f"Obstacle at {min_forward_dist:.2f}m | {self.switch_cooldown}s cooldown"
        #             )
                
        #         # Automatic switch back after cooldown
        #         elif hasattr(self, 'switch_start_time'):
        #             elapsed_time = (current_time - self.switch_start_time).nanoseconds / 1e9
        #             if elapsed_time >= self.switch_cooldown:
        #                 self.current_waypoints = 1
        #             self.get_logger().info(f"Cooldown expired - Returning to primary lane")
        
        # # 5) Default to primary path outside overtake zones
        # else:
        #     if self.current_waypoints != 1:
        #         self.current_waypoints = 1
        #         if hasattr(self, 'switch_start_time'):
        #             del self.switch_start_time
        #         self.get_logger().info("Exiting overtake zone - reset to primary path")




        # 6) Find lookahead waypoint based on current lane
        if self.current_waypoints == 1:
            goal_x, goal_y = self.get_goal_waypoint(car_x, car_y, L, self.kd_tree, self.waypoints)
        else:
            goal_x, goal_y = self.get_goal_waypoint(car_x, car_y, L, self.kd_tree2, self.waypoints2)
            
        if goal_x is None or goal_y is None:
            return  # no valid waypoint found
        
        # Publish the goal waypoint as PointStamped
        point_msg = PointStamped()
        point_msg.header.frame_id = "map"
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.point.x = goal_x
        point_msg.point.y = goal_y
        point_msg.point.z = 0.0

        
        self.waypoint_pub.publish(point_msg)

        # 7) Transform goal to vehicle frame
        goal_y_vehicle = self.translate_point(
            np.array([car_x, car_y]), 
            np.array([goal_x, goal_y])
        )[1]

        # 8) Compute steering from curvature
        curvature = 2.0 * goal_y_vehicle / (L ** 2)
        steering_angle = self.default_kp * curvature

        speed_multiplier = 1.0 - kv * np.abs(steering_angle)
        speed = speed * speed_multiplier


        
        # if current_box == "Box1":
        speed, gap_follow = self.lidar_braking_logic(speed, L, current_box)

        # gap_follow = False
        if gap_follow:
            self.run_gap_follow = True
        else:
            self.run_gap_follow = False        

            # 10) Publish drive command
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = steering_angle
            self.drive_pub.publish(drive_msg)

            # Save previous speed and steering angle
            self.prev_speed = speed
            self.prev_steering_angle = steering_angle


        # 11) Visualize lookahead
        self.publish_lookahead_marker(car_x, car_y, L)



    
    def get_min_forward_distance(self):
        """
        Get the minimum distance from LIDAR in forward-facing direction
        """
        if self.latest_scan is None:
            return float('inf')  # No data yet, return "infinite" distance

        ranges = np.array(self.latest_scan.ranges)
        # Apply a moving average filter to smooth the LIDAR ranges
        window = 5
        if len(ranges) >= window:
            kernel = np.ones(window) / window
            ranges = np.convolve(ranges, kernel, mode='same')
        num_points = len(ranges)
        if num_points == 0:
            return float('inf')

        center_idx = num_points // 2
        window_size = 15  # look +/- 75 samples around the center
        start_idx = max(0, center_idx - window_size)
        end_idx = min(num_points, center_idx + window_size)

        forward_ranges = ranges[start_idx:end_idx]
        valid_ranges = forward_ranges[forward_ranges > 0.08]  # ignore <0.08

        if valid_ranges.size == 0:
            return float('inf')

        return np.min(valid_ranges)
    




    def lidar_braking_logic(self, current_speed, looah, current_box):
        """Calculate speed adjustment based on Lidar scan data.
        
        Args:
            current_speed (float): The current planned speed before braking logic
            
        Returns:
            float: Adjusted speed based on obstacle proximity
        """
        min_forward_dist = self.get_min_forward_distance()
        new_speed = current_speed
        gap_follow = False

        gap_follow_boxes = ["Box1", "Box2", "Box3", "Box4", "Box5", "Box7", "Box8", "Box9", "Box13", "Box14"]

        # Braking logic with clear priority levels
        if min_forward_dist < 0.5:
            new_speed = 0.0
            self.get_logger().warning("EMERGENCY STOP: Obstacle < 0.5m", throttle_duration_sec=0.8)
    
        elif min_forward_dist < looah - 1.0:
        
            self.get_logger().info(f"Caution: Obstacle < 1.0m, limiting to 1.0m/s", throttle_duration_sec=0.8)
            gap_follow = True

            if current_box in gap_follow_boxes:
                new_speed = current_speed
                gap_follow = True
            else:
                new_speed = min(current_speed, 1.0)
                gap_follow = False


        elif min_forward_dist < looah:
            self.get_logger().info(f"Warning: Obstacle < 2.0m, limiting to 2.0m/s", throttle_duration_sec=0.8)
            if current_box in gap_follow_boxes:
                new_speed = current_speed
                gap_follow = True
            else:
                new_speed = min(current_speed, 2.0)
                gap_follow = False


        return new_speed, gap_follow
    
    def publish_lookahead_marker(self, car_x, car_y, L):
        """
        Publish a translucent sphere to represent the lookahead distance around the car.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_circle"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Center the circle at the car's position
        marker.pose.position.x = car_x
        marker.pose.position.y = car_y
        marker.pose.position.z = 0.0

        # Sphere scale.x/scale.y = diameter => 2 * radius
        marker.scale.x = 2.0 * L
        marker.scale.y = 2.0 * L
        marker.scale.z = 0.01  # keep it thin

        # Slightly transparent color (cyan)
        marker.color.a = 0.3
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        self.lookahead_marker_pub.publish(marker)

    def get_goal_waypoint(self, car_x, car_y, look_ahead, kd_tree, waypoints):
        """
        Find a waypoint at least 'look_ahead' meters from (car_x, car_y).
        Returns (goal_x, goal_y) or (None, None) if not found.
        """
        _, idx = kd_tree.query([car_x, car_y])
        # Search forward from idx
        for i in range(idx, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - np.array([car_x, car_y]))
            if dist >= look_ahead:
                return waypoints[i][0], waypoints[i][1]

        # Wrap around if necessary
        for i in range(idx):
            dist = np.linalg.norm(waypoints[i] - np.array([car_x, car_y]))
            if dist >= look_ahead:
                return waypoints[i][0], waypoints[i][1]

        return None, None

    def is_point_in_box(self, x, y, corners):
        """
        corners = [top-left, top-right, bottom-right, bottom-left]
        We'll do a simple bounding-box check.
        """
        point = (x, y)
        polygon = Polygon(corners)
        return polygon.contains(Point(point))

    def create_point(self, x, y, z):
        from geometry_msgs.msg import Point
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        return p

    def translate_point(self, currPoint, targetPoint):
        """
        Transform 'targetPoint' from the map frame to the local vehicle frame,
        using the vehicle's rotation at 'currPoint'.
        """
        H = np.zeros((4, 4))
        # rotation world->vehicle
        H[0:3, 0:3] = np.linalg.inv(self.rot)
        H[0, 3] = currPoint[0]
        H[1, 3] = currPoint[1]
        H[3, 3] = 1.0

        dir_vec = targetPoint - currPoint
        # We multiply by [dx, dy, 0, 0] to effectively do R^-1*(target - current)
        translated_point = (H @ np.array([dir_vec[0], dir_vec[1], 0, 0])).reshape((4))
        return translated_point

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit with Overtaking Logic + Visualization")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
