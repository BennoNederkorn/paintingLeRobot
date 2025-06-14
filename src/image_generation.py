import numpy as np
import config

def generate_image(prompt : str, image : np.ndarray ) -> np.ndarray:
    """
    This function generates a image with dimensions 105mmx148mm
    prompt: string which is the instruction for the image
    image: can be ignored for now. Will be needed if we want to update the old image during drawing.
    """
    return np.random.rand(1050, 1480, 3)

def create_segmentated_image(image : np.ndarray) -> np.ndarray:
    """
    This function creates an image with the same size but only with len(config.colors) colors.
    return a image with len(config.colors) colors with the same size.
    You can also scale the image down
    """
    # TODO use config.colors for the number of colors/pens
    return np.random.rand(105, 148)

def create_color_maps(segmentated_image : np.ndarray) -> list[np.ndarray]:
    """
    This function uses the segmentated_image to create len(config.colors) color maps
    """
    # TODO use config.colors for the number of colors/pens
    return []
