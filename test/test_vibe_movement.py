#!/usr/bin/env python

from lerobot.common.robots.so101_follower.so101_follower import SO101Follower
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.motors import MotorCalibration, Motor, MotorNormMode
import time
import json

def send_action_to_follower():

    # Create configuration - you may need to adjust these parameters
    config = SO101FollowerConfig(
        port="COM7",  # Adjust to your actual port
        use_degrees=True,     # Set to False if you want to use range -100 to 100
        max_relative_target=None,  # Set a value like 30 if you want to limit movement speed
        disable_torque_on_disconnect=True,
        cameras={},  # Empty if no cameras needed for this action
    )
    
    # Load calibration
    with open(r"\\wsl.localhost\Ubuntu\home\zhuolelee\paintingLeRobot\utils\joint_calibration.json", "r") as f:
        calibration_data = json.load(f)
    
    # Convert JSON calibration data to MotorCalibration objects
    calibration_dict = {}
    for motor_name, motor_data in calibration_data.items():
        calibration_dict[motor_name] = MotorCalibration(
            id=motor_data["id"],
            drive_mode=motor_data["drive_mode"],
            homing_offset=motor_data["homing_offset"],
            range_min=motor_data["range_min"],
            range_max=motor_data["range_max"]
        )

    # Create robot instance
    robot = SO101Follower(config)
    
    # Apply the calibration data to the robot
    robot.calibration = calibration_dict
    robot.bus.calibration = calibration_dict
    # Override the motors with custom configuration using calibration data
    # norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
    # robot.bus.motors = {
    #     "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
    #     "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
    #     "elbow_flex": Motor(3, "sts3215", norm_mode_body),
    #     "wrist_flex": Motor(4, "sts3215", norm_mode_body),
    #     "wrist_roll": Motor(5, "sts3215", norm_mode_body),
    #     "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    # }
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
        robot.connect(calibrate=False)  # Set to False to avoid automatic calibration reading
        print("Connected successfully!")
        
        # Write our custom calibration data to the motors
        robot.bus.write_calibration(calibration_dict)
        print("Custom calibration data written to motors!")
        
        # Manually set the calibration as "completed" since we're using our own calibration data
        robot.bus._is_calibrated = True
        
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