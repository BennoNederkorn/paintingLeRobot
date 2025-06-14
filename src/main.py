import numpy as np

import speech_processing 
import image_generation
from robot_controller import RobotController 

def main():
    robot_controller = RobotController()

    robot_controller.start_control_loop() # starts Thread

    while(not robot_controller.has_robot_finished_image()): 
        new_speach_prompt : list[float] = speech_processing.execute_recording()
        if new_speach_prompt == []:
            # continue
            pass

        new_text_prompt : str = speech_processing.speech_to_text(new_speach_prompt)
        painted_regions = robot_controller.get_painted_regions()
        image : np.ndarray = image_generation.generate_image(new_text_prompt, painted_regions)
        segmentated_image : np.ndarray = image_generation.create_segmentated_image(image)
        robot_controller.generate_path(segmentated_image)
        robot_controller.draw()

    robot_controller.stop_control_loop()




if __name__ == "__main__":
    main()