#!/usr/bin/env python

from lerobot.common.robots.so101_follower.so101_follower import SO101Follower
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.motors import MotorCalibration
import time
import json

def send_action_to_follower():

    # Create configuration - you may need to adjust these parameters
    config = SO101FollowerConfig(
        port="/dev/ttyACM0",  # Adjust to your actual port
        use_degrees=True,     # Set to False if you want to use range -100 to 100
        max_relative_target=None,  # Set a value like 30 if you want to limit movement speed
        disable_torque_on_disconnect=True,
        cameras={},  # Empty if no cameras needed for this action
    )


    # Load calibration
    with open("utils/joint_calibration.json", "r") as f:
        calibration_data = json.load(f)
    # print(f"calibration: {calibration_data}")
    # config.calibration = calibration_data

    # Create robot instance
    robot = SO101Follower(config)
    robot.bus.calibration : dict[str, MotorCalibration] = calibration_data
    
    # calibration
    # class MotorCalibration:
    #     id: int
    #     drive_mode: int
    #     homing_offset: int
    #     range_min: int
    #     range_max: int


    try:
        # Connect to the robot
        print("Connecting to SO101 Follower...")
        robot.connect(calibrate=True)  # Set to True if you need calibration
        print("Connected successfully!")
        
        # Define your specific action
        # Each joint position should be in degrees (if use_degrees=True) or -100 to 100 range
        action = {
            "shoulder_pan.pos": 0,    # Adjust these values as needed
            "shoulder_lift.pos": 100,
            "elbow_flex.pos": -100,
            "wrist_flex.pos": 0,
            "wrist_roll.pos": 0,
            "gripper.pos": 0,        # Gripper is always 0-100 range

        }
        
        print(f"Sending action: {action}")
        
        # Send the action
        actual_action = robot.send_action(action)
        print(f"Action sent: {actual_action}")
        
        # Wait for movement to complete (adjust time as needed)
        time.sleep(10.0)
        
        # Optionally, read current position to verify
        current_obs = robot.get_observation()
        motor_positions = {k: v for k, v in current_obs.items() if k.endswith('.pos')}
        print(f"Current positions: {motor_positions}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Always disconnect
        if robot.is_connected:
            print("Disconnecting...")
            robot.disconnect()
            print("Disconnected successfully!")

if __name__ == "__main__":
    send_action_to_follower()