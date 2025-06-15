# test/test_ikpy.py

import os
import numpy as np
from ikpy.chain import Chain

def main():
    # 1) Locate this script’s directory and build the URDF path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    urdf_path = os.path.abspath(os.path.join(script_dir, '..', 'utils', 'robot.urdf'))

    # 2) Load the kinematic chain from the URDF, specifying the base link
    robot_chain = Chain.from_urdf_file(
        urdf_path,
        base_elements=['base']
    )

    # 3) Define the target end-effector position (x, y, z)
    target_position = [0.2, 0.1, 0.0]  # meters

    # 4) Compute inverse kinematics
    #    Supply the 3-element target_position to the IK function
    ik_solution = robot_chain.inverse_kinematics(target_position)

    # 5) Print out the first 5 actuated joint angles (skip fixed base link)
    print("Target joint angles from ikpy:")
    for idx, angle in enumerate(ik_solution[1:6], start=1):
        print(f"  Joint {idx}: {angle:.3f} rad  →  {np.degrees(angle):.2f}°")

if __name__ == '__main__':
    main()
