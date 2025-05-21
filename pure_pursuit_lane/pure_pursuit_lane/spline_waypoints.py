import csv
import numpy as np
from scipy.interpolate import CubicSpline


def load_waypoints(waypoint_file_path, spline_num=500):
    waypoints = []
    try:
        with open(waypoint_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # next(reader) # Skip the header
            for row in reader:
                x = float(row[0])
                y = float(row[1])
                # yaw = float(row[2])
                # qw = float(row[3])
                # qx = float(row[4])
                # qy = float(row[5])
                # qz = float(row[6])
                # waypoints.append([x, y, yaw, qw, qx, qy, qz])
                waypoints.append([x, y])
        waypoints = np.array(waypoints)
        waypoints = spline_interpolate(waypoints, spline_num)

    except Exception as e:
        print(f"Failed to read waypoints: {e}")
        exit()

    return waypoints

def spline_interpolate(waypoints, spline_num):
    """
    Interpolate the waypoints using cubic splines
    :param waypoints: numpy array of waypoints
    :return: numpy array of interpolated waypoints
    """

    x = waypoints[:, 0]
    y = waypoints[:, 1]

    t = np.cumsum(np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1)))
    t = np.insert(t, 0, 0)  

    epsilon = 1e-6
    for i in range(1, len(t)):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + epsilon

    # Cubic spline interpolation
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)

    # Generate new waypoints
    t_new = np.linspace(0, t[-1], num=spline_num)  # 100 points for smoothness
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    waypoints = np.column_stack((x_new, y_new))  # Add yaw and quaternion values as zeros

    return waypoints



if __name__ == "__main__":
    input_file = "/home/shreyas/Documents/ESE6150_F1_Tenth/final-race-roboracer/src/ICRA_Roboracer/pure_pursuit_lane/config/manual_wp.csv"
    output_file = "/home/shreyas/Documents/ESE6150_F1_Tenth/final-race-roboracer/src/ICRA_Roboracer/pure_pursuit_lane/config/manual_wp_spline.csv"

    waypoints = load_waypoints(input_file)

    spline_wp = spline_interpolate(waypoints, 600)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['x', 'y'])
        for wp in spline_wp:
            writer.writerow([wp[0], wp[1]])
    print(f"Interpolated waypoints saved to {output_file}")
    print("Done")