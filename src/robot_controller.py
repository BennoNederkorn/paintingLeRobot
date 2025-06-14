import numpy as np
import config

class RobotController:
    # def __init__(self):
    #     pass

    def update_color_maps(self, color_maps : list[np.ndarray]):
        self.color_maps = color_maps

    def start_control_loop(self, color_maps : list[np.ndarray]):
        """
        This starts the robot. The robot is now waiting for additional commands.
        """
        self.color_maps = color_maps

        for i in range(len(color_maps)):
            waypoints : list[list[int]] = self.generate_path(color_maps[i])
            self.grab_pen(i)
            self.execute_path(waypoints)


    
    # def stop_control_loop(self):
    #     """
    #     This starts the robot. The robot is now waiting for additional commands.
    #     """

    def generate_path(self, color_map : np.ndarray) -> list[list[int]]:
        """
        this function generates a path on which the robot has to move to draw
        """
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