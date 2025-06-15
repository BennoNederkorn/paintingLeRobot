import numpy as np
import config
import math 
import json
import os
from ikpy.chain import Chain


class RobotController:
    def __init__(self):
        # 1) Locate the URDF file relative to this script
        script_dir = os.path.dirname(os.path.realpath(__file__))
        urdf_path  = os.path.abspath(os.path.join(script_dir, '..', 'utils', 'robot.urdf'))

        # 2) Build the kinematic chain from the URDF, specifying the base link
        self.robot_chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=['base']
        )

        with open("utils/joint_calibration.json", "r") as f:
            data = json.load(f)

    def start_control_loop(self, waypoints : dict[str, list[list[tuple[float, float]]]]):
        """
        This starts the robot. The robot is now waiting for additional commands.
        """
        self.waypoints = waypoints

        for color, waypoint_listlist in  self.waypoints:
            # grap pen
            for waypoint_list in waypoint_listlist:
                # move to all the points waypoint_list
                for waypoint in waypoint_list:
                    self.move_to_waypoint(waypoint)
                # rise pen
            # put pen back

    def move_to_waypoint(self, waypoint : tuple[float, float]):
        waypoint_3D : tuple[float, float, float] = self.twoD_into_threeD(waypoint)
        self.waypoint_3D_to_morter_values(waypoint_3D)

    def twoD_into_threeD(self, waypoint : tuple[float, float]):
        pass

    def waypoint_3D_to_morter_values(self, waypoint_3D : tuple[float, float, float]):
        pass

    def grab_pen(self, i : int):
        """
        this function uses the robot to pick a pen from the pen holder
        """
        pass

    def execute_path(self, waypoints : list[list[int]]):
        """
        This function starts drawing the given image.
        """
        pass

    # def has_robot_finished_image(self):
    #     """
    #     This function checks if the robot has finished the whole image.
    #     """
    #     return False
    
    # def get_painted_regions(self):
    #     return None

    def create_calibration_map(self, angle_start, value_start, angle_end, value_end):
        """
        Returns a function mapping a joint angle (radians) to a raw command value via linear interpolation.
        """
        slope = (value_end - value_start) / (angle_end - angle_start)
        intercept = value_start - slope * angle_start

        def map_angle(angle: float) -> int:
            raw = slope * angle + intercept
            return int(round(raw))

        return map_angle

    def compute_raw_commands(self, waypoint):
        """
        Given a 2D waypoint [x, y], compute IK and map to raw motor commands.
        Returns a dict of {joint_name: raw_value}.
        """
        x, y = waypoint
        # 3) Solve IK for [x, y, 0]
        ik_solution = self.robot_chain.inverse_kinematics([x, y, 0.0])

        # 4) Map angles to raw commands
        raw_commands = {}
        for idx, name in enumerate(joint_names, start=1):
            angle = ik_solution[idx]
            raw_commands[name] = joint_calibrations[name](angle)
        return raw_commands


if __name__ == "__main__":
    # Calibration maps for each joint (example values)
    robotController = RobotController()
    


    joint_calibrations = {}
    action = {}
            # "shoulder_pan.pos": 0,    # Adjust these values as needed
            # "shoulder_lift.pos": 50,
            # "elbow_flex.pos": -50,
            # "wrist_flex.pos": 0,
            # "wrist_roll.pos": 0,
            # "gripper.pos": 0,        # Gripper is always 0-100 range

    for joint_name, values in data.items():
        range_min = values.get("range_min")
        range_max = values.get("range_max")
        joint_calibrations[joint_name]=robotController.create_calibration_map(0.0, range_min, math.pi, range_max)

    # Example waypoints to test
    waypoints = [
        [0.2, 0.1],
        [0.15, 0.05],
        [0.1, 0.1]
    ]
    
    # for joint_name, values in data.items():
    #     action[joint_name.pos] = joint_calibrations[]()

    for wp in waypoints:
        raw_cmds = robotController.compute_raw_commands(wp)
        print(f"Waypoint {wp}: Raw commands → {raw_cmds}")

    
