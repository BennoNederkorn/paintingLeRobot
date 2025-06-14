import numpy as np

import RobotController from robot_controller

def execute_recording() -> list[float]:
    """
    This function strarts a recording while a key is pressed and returns a .wav file/data
    """
    return []


def speech_to_text(data : list[float]) -> str:
    """
    This function gets a speach data and returns text
    """
    
    new_prompt : str = "Hello Lee" 
    print(new_prompt)
    return new_prompt

def generate_image(prompt : str, image : np.ndarray ) -> np.ndarray:

    return np.random.rand(255, 255, 3)

def create_segmentated_image(image : np.ndarray) -> np.ndarray:
    return np.random.rand(255, 255, 3)


def main():
    robot_controller = RobotController()

    robot_controller.start_control_loop() # starts Thread

    while(robot_controller.has_robot_finished_image()):
        new_speach_prompt : list[float] = execute_recording()
        new_text_prompt : str = speech_to_text(new_speach_prompt)
        painted_regions = robot_controller.get_painted_regions()
        image : np.ndarray = generate_image(new_text_prompt)
        segmentated_image : np.ndarray = create_segmentated_image(image)

    robot_controller.stop_control_loop()




if __name__ == "__main__":
    main()