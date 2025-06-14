import numpy as np

import src.speech_processing as speech_processing 
from robot_controller import RobotController 

def generate_image(prompt : str, image : np.ndarray ) -> np.ndarray:

    return np.random.rand(255, 255, 3)

def create_segmentated_image(image : np.ndarray) -> np.ndarray:
    return np.random.rand(255, 255, 3)


def main():
    robot_controller = RobotController()

    robot_controller.start_control_loop() # starts Thread

    while(robot_controller.has_robot_finished_image()):
        new_speach_prompt : list[float] = speech_processing.execute_recording()
        new_text_prompt : str = speech_processing.speech_to_text(new_speach_prompt)
        painted_regions = robot_controller.get_painted_regions()
        image : np.ndarray = generate_image(new_text_prompt, painted_regions)
        segmentated_image : np.ndarray = create_segmentated_image(image)
        # robot_controller.

    robot_controller.stop_control_loop()




if __name__ == "__main__":
    main()