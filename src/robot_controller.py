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

        for color_map in color_maps:
            waypoints : list[list[int]] = generate_path(color_map)
            # grab right pen
            # pass waypoints to motion planner 

    
    # def stop_control_loop(self):
    #     """
    #     This starts the robot. The robot is now waiting for additional commands.
    #     """

    def generate_path(self, color_map : np.ndarray) -> list[list[int]]:


    def start_drawing(self):
        """
        This function starts drawing the given image. This function is calling all the ne
        """
        generate_path_plan()


    def has_robot_finished_image(self):
        """
        This function checks if the robot has finished the whole image.
        """
        return False
    
    def get_painted_regions(self):
        return None
    
    def generate_path(self, segmentated_image : np.ndarray):
        """
        This function plans the path
        """

    def draw(self):

    