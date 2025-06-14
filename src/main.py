import numpy as np
from typing import Optional
from pathlib import Path


import speech_processing 
import image_generation
from robot_controller import RobotController 

def main():
    robot_controller = RobotController()

    # robot_controller.start_control_loop() # starts Thread
    # while(not robot_controller.has_robot_finished_image()): # change this later to a while loop for 

    speech_processing.create_wav_data()
    new_text_prompt : str = speech_processing.create_txt_prompt()

    painted_regions = None # robot_controller.get_painted_regions()
    output_file_path : Path = Path.cwd() / f"output.png"
    image : np.ndarray = image_generation.generate_image(prompt=new_text_prompt, output_file_path=output_file_path, image=painted_regions)

    # segmentated_image : np.ndarray = image_generation.create_segmentated_image(image)
    # color_maps : list[np.ndarray] = image_generation.create_color_maps(segmentated_image)
    # robot_controller.start_control_loop(color_maps)




if __name__ == "__main__":
    main()