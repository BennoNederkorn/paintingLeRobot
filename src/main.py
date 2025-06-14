import numpy as np
from typing import Optional

import speech_processing 
import image_generation
from robot_controller import RobotController 

def main():
    robot_controller = RobotController()

    # robot_controller.start_control_loop() # starts Thread
    # while(not robot_controller.has_robot_finished_image()): # change this later to a while loop for 
    
    new_speach_prompt : Optional[list[float]] = speech_processing.execute_recording()
    # TODO check if speech processing failed
    new_text_prompt : str = speech_processing.speech_to_text(new_speach_prompt)
    painted_regions = robot_controller.get_painted_regions()
    image : np.ndarray = image_generation.generate_image(new_text_prompt, painted_regions)
    segmentated_image : np.ndarray = image_generation.create_segmentated_image(image)
    color_maps : list[np.ndarray] = image_generation.create_color_maps(segmentated_image)
    robot_controller.start_control_loop(color_maps)




if __name__ == "__main__":
    main()