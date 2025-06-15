# test/test_ikpy.py

import os
import numpy as np
from ikpy.chain import Chain
import math


def create_calibration_map(angle_start, value_start, angle_end, value_end):
    """
    Returns a function mapping a joint angle (radians) to a raw command value via linear interpolation.
    """
    slope = (value_end - value_start) / (angle_end - angle_start)
    intercept = value_start - slope * angle_start

    def map_angle(angle: float) -> int:
        raw = slope * angle + intercept
        return int(round(raw))

    return map_angle

# Calibration maps for each joint (example values)
joint_calibrations = {
    'shoulder_pan':  create_calibration_map(0.0,       697, math.pi,    3342),
    'shoulder_lift': create_calibration_map(-math.pi/2, 957, math.pi/2, 3361),
    'elbow_flex':    create_calibration_map(0.0,       658, math.pi,    2893),
    'wrist_flex':    create_calibration_map(-math.pi/2, 815, math.pi/2, 3289),
    'wrist_roll':    create_calibration_map(-math.pi,    0,   math.pi,    4095),
}

# 1) Locate the URDF file relative to this script
script_dir = os.path.dirname(os.path.realpath(__file__))
urdf_path  = os.path.abspath(os.path.join(script_dir, '..', 'utils', 'robot.urdf'))

# 2) Build the kinematic chain from the URDF, specifying the base link
robot_chain = Chain.from_urdf_file(
    urdf_path,
    base_elements=['base']
)

# Joint names corresponding to the first 5 actuated joints in the chain
joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']


def compute_raw_commands(waypoint):
    """
    Given a 2D waypoint [x, y], compute IK and map to raw motor commands.
    Returns a dict of {joint_name: raw_value}.
    """
    x, y = waypoint
    # 3) Solve IK for [x, y, 0]
    ik_solution = robot_chain.inverse_kinematics([x, y, 0.0])

    # 4) Map angles to raw commands
    raw_commands = {}
    for idx, name in enumerate(joint_names, start=1):
        angle = ik_solution[idx]
        raw_commands[name] = joint_calibrations[name](angle)
    return raw_commands
'''
def main():
    # Example waypoints to test
    waypoints = [
        [0.2, 0.1],
        [0.15, 0.05],
        [0.1, 0.1]
    ]
    for wp in waypoints:
        raw_cmds = compute_raw_commands(wp)
        print(f"Waypoint {wp}: Raw commands → {raw_cmds}")


if __name__ == '__main__':
    main()




'''


