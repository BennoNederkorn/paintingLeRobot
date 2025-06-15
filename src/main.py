import numpy as np
from typing import Optional
from pathlib import Path
from config import PALETTE
import os

import speech_processing 
import image_generation
import color_quantization
from robot_controller import RobotController 
import layers_to_waypoints

def main():
    robot_controller = RobotController()

    # robot_controller.start_control_loop() # starts Thread
    # while(not robot_controller.has_robot_finished_image()): # change this later to a while loop for 

    # speech_processing.create_wav_data()
    # new_text_prompt : str = speech_processing.create_txt_prompt()

    # painted_regions = None # robot_controller.get_painted_regions()
    input_folder = Path.cwd() / f"input_folder"
    output_folder = Path.cwd() / f"output_folder"
    # output_file_path : Path = input_folder / f"output.png"
    # image : np.ndarray = image_generation.generate_image(prompt=new_text_prompt, output_file_path=output_file_path, image=painted_regions)
    # color_quantization.process_and_downsample(input_folder, output_folder)


    file_path1 = 'output_folder/output_quantized.png'
    file_path2 = 'output_folder/output_small.png'
    if os.path.exists(file_path1):
        os.remove(file_path1)
    if os.path.exists(file_path2):
        os.remove(file_path2)

    waypoints : dict[str, list[list[tuple[float, float]]]] = layers_to_waypoints.generate_fill_with_overlay(
        layer_dir=str(output_folder),
        small_dir="image_maps/output",
        output_dir="image_maps/filled_regions_overlay",
        palette=PALETTE,
        paper_size_mm=(105.0,148.0),
        grid_size_px=(26,38),
        origin_mm=(0.0,0.0),
    )
    print(waypoints)

    # robot_controller.start_control_loop(waypoints)




if __name__ == "__main__":
    main()